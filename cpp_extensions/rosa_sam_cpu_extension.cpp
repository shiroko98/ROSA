#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using PredictResult = std::tuple<std::vector<int64_t>, std::vector<int64_t>>;

// Dynamic suffix-link tree. Each token assigns its end position to the entire
// root-to-last path. Eagerly walking that path is quadratic for repeated tokens.
// A rooted link-cut tree performs the SAME assignment and point queries lazily,
// including SAM clone reparenting, in amortized logarithmic time per operation.
// No token (padding/special included) is dropped and there is no future lookup.
class LatestEndposTree {
    struct Node {
        int64_t child[2] = {-1, -1};
        int64_t parent = -1;
        int64_t value = -1;
        int64_t pending = -1;
    };
    std::vector<Node> nodes;
    std::vector<int64_t> ancestors;

    bool auxiliary_root(int64_t x) const {
        const auto p = nodes[x].parent;
        return p == -1 || (nodes[p].child[0] != x && nodes[p].child[1] != x);
    }
    void assign(int64_t x, int64_t value) {
        if (x == -1) return;
        nodes[x].value = value;
        nodes[x].pending = value;
    }
    void push(int64_t x) {
        if (nodes[x].pending < 0) return;
        assign(nodes[x].child[0], nodes[x].pending);
        assign(nodes[x].child[1], nodes[x].pending);
        nodes[x].pending = -1;
    }
    void rotate(int64_t x) {
        const auto p = nodes[x].parent;
        const auto g = nodes[p].parent;
        const int side = nodes[p].child[1] == x;
        const auto middle = nodes[x].child[side ^ 1];
        if (!auxiliary_root(p)) nodes[g].child[nodes[g].child[1] == p] = x;
        nodes[x].parent = g;
        nodes[x].child[side ^ 1] = p;
        nodes[p].parent = x;
        nodes[p].child[side] = middle;
        if (middle != -1) nodes[middle].parent = p;
    }
    void splay(int64_t x) {
        ancestors.clear();
        auto y = x;
        ancestors.push_back(y);
        while (!auxiliary_root(y)) {
            y = nodes[y].parent;
            ancestors.push_back(y);
        }
        for (auto it = ancestors.rbegin(); it != ancestors.rend(); ++it) push(*it);
        while (!auxiliary_root(x)) {
            const auto p = nodes[x].parent;
            const auto g = nodes[p].parent;
            if (!auxiliary_root(p)) {
                const bool aligned = (nodes[p].child[1] == x) == (nodes[g].child[1] == p);
                rotate(aligned ? p : x);
            }
            rotate(x);
        }
    }
    void access(int64_t x) {
        int64_t previous = -1;
        for (auto y = x; y != -1; y = nodes[y].parent) {
            splay(y);
            nodes[y].child[1] = previous;
            // The detached right child's parent remains its path-parent.
            if (previous != -1) nodes[previous].parent = y;
            previous = y;
        }
        splay(x);
    }
public:
    explicit LatestEndposTree(int64_t capacity) : nodes(capacity) {}
    void grow(int64_t size) { if (size > static_cast<int64_t>(nodes.size())) nodes.resize(size); }
    int64_t get(int64_t x) {
        access(x);
        return nodes[x].value;
    }
    void initialize(int64_t x, int64_t value = -1) { nodes[x].value = value; }
    // Child is a represented-tree root. Parent does not need to be exposed.
    void link(int64_t child, int64_t parent) { nodes[child].parent = parent; }
    void cut_parent(int64_t child) {
        access(child);
        const auto left = nodes[child].child[0];
        if (left != -1) nodes[left].parent = -1;
        nodes[child].child[0] = -1;
    }
    void assign_ancestors(int64_t x, int64_t value) {
        access(x);
        assign(x, value);
    }
};

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
    LatestEndposTree endpos(state_cap);

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
                endpos.initialize(u, endpos.get(q));
                // Clone inherits q's former suffix parent and latest occurrence.
                endpos.link(u, link[static_cast<size_t>(u)]);
                endpos.cut_parent(q);
                endpos.link(q, u);

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

        endpos.link(r, link[static_cast<size_t>(r)]);
        // r is new. Its suffix parent (if non-root) is the longest existing
        // repeated suffix and already has an occurrence strictly before i.
        const auto v = link[static_cast<size_t>(r)];
        int64_t addr_id = -1;
        int64_t best_match = length[static_cast<size_t>(v)];
        if (best_match > 0 && best_match >= min_match_len) {
            const int64_t idx = endpos.get(v) + 1;
            if (0 <= idx && idx <= i) addr_id = seq[static_cast<size_t>(idx)];
        }
        pred[static_cast<size_t>(i)] = addr_id;
        match_len[static_cast<size_t>(i)] = best_match;
        last = r;
        endpos.assign_ancestors(last, i);
    }

    return std::make_tuple(pred, match_len);
}

// Persistent version of the same suffix-link/LCT algorithm. No full-history replay.
class IncrementalSAM {
    std::vector<std::unordered_map<int64_t,int64_t>> trans{1};
    std::vector<int64_t> link{-1},length{0},seq;
    LatestEndposTree endpos{1};
    int64_t last=0,min_match_len=1;
    int64_t new_state(){
        int64_t id=trans.size();trans.emplace_back();link.push_back(-1);length.push_back(0);
        endpos.grow(id+1);return id;
    }
public:
    explicit IncrementalSAM(int64_t min_match=1):min_match_len(min_match){}
    std::tuple<int64_t,int64_t> update_one(int64_t token){
        const int64_t i=seq.size();seq.push_back(token);
        const int64_t r = new_state();
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
                const int64_t u = new_state();
                trans[static_cast<size_t>(u)] = trans[static_cast<size_t>(q)];
                length[static_cast<size_t>(u)] = length[static_cast<size_t>(p)] + 1;
                link[static_cast<size_t>(u)] = link[static_cast<size_t>(q)];
                endpos.initialize(u, endpos.get(q));
                // Clone inherits q's former suffix parent and latest occurrence.
                endpos.link(u, link[static_cast<size_t>(u)]);
                endpos.cut_parent(q);
                endpos.link(q, u);

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

        endpos.link(r, link[static_cast<size_t>(r)]);
        // r is new. Its suffix parent (if non-root) is the longest existing
        // repeated suffix and already has an occurrence strictly before i.
        const auto v = link[static_cast<size_t>(r)];
        int64_t addr_id = -1;
        int64_t best_match = length[static_cast<size_t>(v)];
        if (best_match > 0 && best_match >= min_match_len) {
            const int64_t idx = endpos.get(v) + 1;
            if (0 <= idx && idx <= i) addr_id = seq[static_cast<size_t>(idx)];
        }

        last=r;endpos.assign_ancestors(last,i);
        return {addr_id,best_match};
    }
    PredictResult prefill(const std::vector<int64_t>& tokens){
        std::vector<int64_t> p,m;p.reserve(tokens.size());m.reserve(tokens.size());
        for(auto token:tokens){auto [a,l]=update_one(token);p.push_back(a);m.push_back(l);}
        return {p,m};
    }
    int64_t size() const{return seq.size();}
    IncrementalSAM clone() const{return *this;}
};

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
    pybind11::class_<IncrementalSAM>(m,"IncrementalSAM")
        .def(pybind11::init<int64_t>(),pybind11::arg("min_match_len")=1)
        .def("update_one",&IncrementalSAM::update_one)
        .def("prefill",&IncrementalSAM::prefill)
        .def("clone",&IncrementalSAM::clone)
        .def("size",&IncrementalSAM::size);
    m.attr("algorithm_version") = "suffix_link_lct_v1";
    m.def("sam_rosa_predict", &sam_rosa_predict, "Compiled CPU SAM predictor");
    m.def("sam_rosa_predict_many", &sam_rosa_predict_many, "Compiled CPU SAM predictor for many sequences");
}
