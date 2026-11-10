// Greedy byte-trie tokenizer. IDs >=65530 are intentionally excluded: they are
// private ChatML additions, not trained text tokens in the supplied G1 model.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace py = pybind11;
struct Node {
    std::array<int32_t, 256> next;
    int32_t token = -1;
    Node() { next.fill(-1); }
};
class Tokenizer {
    std::vector<Node> nodes;
public:
    explicit Tokenizer(const std::vector<std::pair<py::bytes, int>>& vocabulary) {
        nodes.emplace_back();
        for (const auto& item : vocabulary) {
            if (item.second <= 0 || item.second >= 65530) continue;
            const std::string word = item.first;
            int32_t current = 0;
            for (unsigned char byte : word) {
                int32_t next = nodes[current].next[byte];
                if (next < 0) {
                    next = static_cast<int32_t>(nodes.size());
                    nodes[current].next[byte] = next;
                    nodes.emplace_back();
                }
                current = next;
            }
            nodes[current].token = item.second;
        }
    }
    py::tuple encode(const py::bytes& input, size_t train_start, size_t train_end) const {
        const std::string text = input;
        if (train_start > train_end || train_end > text.size())
            throw std::invalid_argument("Invalid supervised byte interval");
        std::string ids, mask;
        ids.reserve(text.size()); mask.reserve(text.size() / 2);
        size_t start = 0;
        while (start < text.size()) {
            int32_t current = 0, token = -1;
            size_t end = start;
            for (size_t cursor = start; cursor < text.size(); ++cursor) {
                current = nodes[current].next[static_cast<unsigned char>(text[cursor])];
                if (current < 0) break;
                if (nodes[current].token >= 0) {
                    token = nodes[current].token; end = cursor + 1;
                }
            }
            if (token < 0) throw std::runtime_error("Byte not covered by RWKV vocabulary");
            ids.push_back(static_cast<char>(token & 255));
            ids.push_back(static_cast<char>((token >> 8) & 255));
            mask.push_back(end > train_start && start < train_end ? 1 : 0);
            start = end;
        }
        return py::make_tuple(py::bytes(ids), py::bytes(mask));
    }
};
PYBIND11_MODULE(rwkv_plain_tokenizer_ext, module) {
    py::class_<Tokenizer>(module, "Tokenizer")
        .def(py::init<const std::vector<std::pair<py::bytes, int>>&>())
        .def("encode", &Tokenizer::encode);
}
