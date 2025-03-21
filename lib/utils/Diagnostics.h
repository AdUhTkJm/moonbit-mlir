#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "lib/utils/Common.h"

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
    void report() const;

    Severity sev;
    Location from, to;
    std::string msg;
  
    std::string reportSeverity(Severity sev) const;
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
