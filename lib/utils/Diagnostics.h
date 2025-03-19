#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "lib/utils/Common.h"
#include <sstream>
#include <iostream>
#include <cassert>
#include <format>
#include <vector>
#include <cstdlib>
#include <unistd.h>

#define RED     "\033[1;31m"
#define MAGENTA "\033[1;35m"
#define CYAN    "\033[1;36m"
#define RESET   "\033[0m"

namespace mbt {

class Diagnostics {
private:

  class Diagnostic {
  public:
    enum Severity {
      Error, Warning, Info,
    };
  
    Diagnostic(Severity sev, const Location& from, const Location &to, const std::string& msg)
        : sev(sev), from(from), to(to), msg(msg) {}
  
    // Output formatted message
    void report() const {
      auto sev_s = reportSeverity(sev);
      std::cerr << format("{}:{}:{}: {}: {}\n", from.filename.str(), from.line, from.col, sev_s, msg);
    }

    Severity sev;
    Location from, to;
    std::string msg;
  
    std::string reportSeverity(Severity sev) const {
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
  };
  
  static std::vector<Diagnostic> diags;
  static std::vector<std::string> lines;

  static int errorCnt, warningCnt;

public:
  // Record an error.
  static void error(const Location& from, const Location &to, const std::string& msg) {
    errorCnt++;
    diags.push_back(Diagnostic(Diagnostic::Error, from, to, msg));
  }

  // Record a warning.
  static void warning(const Location& from, const Location &to, const std::string& msg) {
    warningCnt++;
    diags.push_back(Diagnostic(Diagnostic::Warning, from, to, msg));
  }

  // Record an informational message.
  static void info(const Location& from, const Location &to, const std::string& msg) {
    diags.push_back(Diagnostic(Diagnostic::Info, from, to, msg));
  }

  // Output all diagnostics.
  static void reportAll(bool exits = true);

  static int errorCount() { return errorCnt; }
  static int warningCount() { return warningCnt; }
  static void setInput(const std::string &input);
};

} // namespace mbt

#endif
