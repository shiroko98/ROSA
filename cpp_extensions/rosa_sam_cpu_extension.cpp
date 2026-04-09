#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using PredictResult = std::tuple<std::vector<int64_t>, std::vector<int64_t>>;

PredictResult sam_rosa_predict_impl(const std::vector<int64_t>& seq, int64_t min_match_len) {
    const int64_t n = static_cast<int64_t>(seq.size());
    std::vector<int64_t> pred(static_cast<size_t>(n), -1);
    std::vector<int64_t> match_len(static_cast<size_t>(n), 0);
    if (n == 0) {
        return std::make_tuple(pred, match_len);
    }

    const int64_t state_cap = 2 * n + 1;
    std::vector<std::unordered_map<int64_t, int64_t>> trans(static_cast<size_t>(state_cap));
    std::vector<int64_t> link(static_cast<size_t>(state_cap), -1);
    std::vector<int64_t> length(static_cast<size_t>(state_cap), 0);
    std::vector<int64_t> endpos(static_cast<size_t>(state_cap), -1);

    int64_t last = 0;
    int64_t z = 1;

    for (int64_t i = 0; i < n; ++i) {
        const int64_t token = seq[static_cast<size_t>(i)];
        const int64_t r = z++;
        length[static_cast<size_t>(r)] = length[static_cast<size_t>(last)] + 1;
        int64_t p = last;

        while (p != -1 && trans[static_cast<size_t>(p)].find(token) == trans[static_cast<size_t>(p)].end()) {
            trans[static_cast<size_t>(p)][token] = r;
            p = link[static_cast<size_t>(p)];
        }

        if (p == -1) {
            link[static_cast<size_t>(r)] = 0;
        } else {
            const int64_t q = trans[static_cast<size_t>(p)][token];
            if (length[static_cast<size_t>(p)] + 1 == length[static_cast<size_t>(q)]) {
                link[static_cast<size_t>(r)] = q;
            } else {
                const int64_t u = z++;
                trans[static_cast<size_t>(u)] = trans[static_cast<size_t>(q)];
                length[static_cast<size_t>(u)] = length[static_cast<size_t>(p)] + 1;
                link[static_cast<size_t>(u)] = link[static_cast<size_t>(q)];
                endpos[static_cast<size_t>(u)] = endpos[static_cast<size_t>(q)];

                while (p != -1) {
                    auto iter = trans[static_cast<size_t>(p)].find(token);
                    if (iter == trans[static_cast<size_t>(p)].end() || iter->second != q) {
                        break;
                    }
                    iter->second = u;
                    p = link[static_cast<size_t>(p)];
                }

                link[static_cast<size_t>(q)] = u;
                link[static_cast<size_t>(r)] = u;
            }
        }

        int64_t v = r;
        int64_t addr_id = -1;
        int64_t best_match = 0;
        while (v != -1) {
            if (length[static_cast<size_t>(v)] > 0 && endpos[static_cast<size_t>(v)] >= 0) {
                best_match = length[static_cast<size_t>(v)];
                if (best_match >= min_match_len) {
                    const int64_t idx = endpos[static_cast<size_t>(v)] + 1;
                    if (0 <= idx && idx < n) {
                        addr_id = seq[static_cast<size_t>(idx)];
                    }
                }
                break;
            }
            v = link[static_cast<size_t>(v)];
        }

        pred[static_cast<size_t>(i)] = addr_id;
        match_len[static_cast<size_t>(i)] = best_match;
        last = r;

        v = last;
        while (v != -1 && endpos[static_cast<size_t>(v)] < i) {
            endpos[static_cast<size_t>(v)] = i;
            v = link[static_cast<size_t>(v)];
        }
    }

    return std::make_tuple(pred, match_len);
}

}  // namespace

PredictResult sam_rosa_predict(const std::vector<int64_t>& seq, int64_t min_match_len) {
    return sam_rosa_predict_impl(seq, min_match_len);
}

std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>> sam_rosa_predict_many(
    const std::vector<std::vector<int64_t>>& sequences,
    int64_t min_match_len
) {
    std::vector<std::vector<int64_t>> all_preds;
    std::vector<std::vector<int64_t>> all_match_lens;
    all_preds.reserve(sequences.size());
    all_match_lens.reserve(sequences.size());

    for (const auto& seq : sequences) {
        auto [pred, match_len] = sam_rosa_predict_impl(seq, min_match_len);
        all_preds.push_back(std::move(pred));
        all_match_lens.push_back(std::move(match_len));
    }
    return std::make_tuple(all_preds, all_match_lens);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sam_rosa_predict", &sam_rosa_predict, "Compiled CPU SAM predictor");
    m.def("sam_rosa_predict_many", &sam_rosa_predict_many, "Compiled CPU SAM predictor for many sequences");
}
