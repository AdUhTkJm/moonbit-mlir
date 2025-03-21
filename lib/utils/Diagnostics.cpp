#include "Diagnostics.h"
#include <unistd.h>
#include <sstream>

#define RED     "\033[1;31m"
#define MAGENTA "\033[1;35m"
#define CYAN    "\033[1;36m"
#define RESET   "\033[0m"

using namespace mbt;

std::vector<Diagnostics::Diagnostic> Diagnostics::diags;
std::vector<std::string> Diagnostics::lines;
int Diagnostics::errorCnt = 0;
int Diagnostics::warningCnt = 0;

std::string Diagnostics::Diagnostic::reportSeverity(Severity sev) const {
  if (isatty(fileno(stdout))) {
    switch (sev) {
    case Severity::Error: return RED "error" RESET;
    case Severity::Warning: return MAGENTA "warning" RESET;
    case Severity::Info: return CYAN "note" RESET;
    }
  } else {
    switch (sev) {
    case Severity::Error: return "error";
    case Severity::Warning: return "warning";
    case Severity::Info: return "note";
    }
  }
  assert(false);
}

void Diagnostics::Diagnostic::report() const {
  auto sev_s = reportSeverity(sev);
  std::cerr << format("{}:{}:{}: {}: {}\n", from.filename.str(), from.line, from.col, sev_s, msg);
}

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
      for (size_t i = 0; i < diag.to.col - diag.from.col; ++i)
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
