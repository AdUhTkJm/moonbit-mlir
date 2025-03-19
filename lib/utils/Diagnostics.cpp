#include "Diagnostics.h"

using namespace mbt;

std::vector<Diagnostics::Diagnostic> Diagnostics::diags;
std::vector<std::string> Diagnostics::lines;
int Diagnostics::errorCnt = 0;
int Diagnostics::warningCnt = 0;

void Diagnostics::reportAll(bool exits) {
  for (const auto& diag : diags) {
    diag.report();

    // This is probably hitting end of file. Nothing to display.
    if (diag.to.line > lines.size())
      continue;

    // When `from.col` is 0 it would wrap around.
    size_t indent_width = diag.from.col - (diag.from.col < 1 ? 0 : 1);
    std::cerr << lines[diag.from.line - 1] << "\n";
    for (size_t i = 0; i < indent_width; ++i)
      std::cerr << ' ';

    // Give up the indicator if the error spans multiple lines
    if (diag.to.line > diag.from.line) {
      for (size_t i = diag.from.line + 1; i < diag.to.line; ++i)
        std::cerr << lines[i - 1] << "\n";
    } else {
      assert(diag.to.col >= diag.from.col);
      for (size_t i = 0; i <= diag.to.col - diag.from.col; ++i)
        std::cerr << '^';
      std::cerr << "\n";
    }
  }
  if (errorCount() > 0 && exits)
    exit(1);
}

void Diagnostics::setInput(const std::string &input) {
  std::istringstream ss(input);
  std::string line;
  while (std::getline(ss, line))
    lines.push_back(line);
}
