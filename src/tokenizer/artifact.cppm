export module spar.tokenizer.artifact;

import std;
export import spar.tokenizer.byte_bpe;

export namespace spar::tokenizer {

/// Saves canonical little-endian SPARTOKN format version 1 through a sibling temporary file.
void save_tokenizer(const std::filesystem::path& path, const ByteBPETokenizer& tokenizer);

/// Loads and validates a standalone SPARTOKN artifact.
[[nodiscard]] ByteBPETokenizer load_tokenizer(const std::filesystem::path& path);

} // namespace spar::tokenizer
