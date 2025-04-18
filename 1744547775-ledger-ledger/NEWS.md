# Ledger NEWS

## 3.4.x (unreleased)

- Update required versions of various dependencies
  (CMake 3.16.2, Boost 1.72.0, Gmp 6.1.2, Mpfr 4.0.2, Python 3.9, doxygen 1.9.5)

- docs: Enable stand-alone building

- Include contrib files in distribution

- Fix related reports when using bucket transactions (ledger/ledger#2220)

- Add support to build ledger with readline

- Add commodity value type in value expressions to resolve aliases

## 3.3.2 (2023-03-30)

- Fix divide by zero (ledger/ledger#777, ledger/ledger#2207)

- Increase string size limit in src/unistring.h assert (ledger/ledger#2174)

- Require tzdata for Nix flake build (ledger/ledger#2213)

## 3.3.1 (2023-03-03)

- Fix regression leading to incorrect error about `format` directives (ledger/ledger#2205)

- Add information about compile features to `--version`

- Fix compiler warnings by minimizing the use of deprecated APIs

- Update flake.nix to match nixpkgs ledger/default.nix

- Remove unused Python server related code

- Various documentation improvements

## 3.3 (2023-02-08)

- Use `$PAGER` when environment variable is set (ledger/ledger#1674)

- Make `--depth` correctly fold postings to accounts of greater depth into the
  parent at the specified level (ledger/ledger#987)

- When using wild-cards in the `include` directive, include matched files in
  sorted order (ledger/ledger#1659)

- Ensure absolute path for include (ledger/ledger#2075)

- Try to use `$XDG_HOME_CONFIG/ledger/ledgerrc` or `~/.config/ledger/ledgerrc`
  first

- Improve Python 3 support and drop support for Python 2

- Add support for automatically reading files encrypted with GPG (ledger/ledger#1949)

- Add support for a "debit" column in the `convert` command (ledger/ledger#1120)

- Fix parsing of files without end of line (ledger/ledger#516)

- Fix incorrect parsing of expressions containing a `-` without spaces (ledger/ledger#2001)

- Fix payee metadata on postings not being validated and payee aliases not
  being honored (ledger/ledger#556, ledger/ledger#1892)

- Fix ledger interpreting a posting with 0 difference as a null-posting,
  which leads to it auto-balancing the posting (ledger/ledger#1942)

- Correctly escape all string values in lisp report (ledger/ledger#2034)

- Fix a regression where empty commodities were shown (ledger/ledger#1969)

- Fix a regression where using multiple commodities in one transaction triggers
  an assertion (ledger/ledger#1998)

- Fix --time-colon for negative time amounts

- Use correct int return type for stream input operations (ledger/ledger#2058)

- Use amount_width for balance report

- Remove some UTF-8 code that was having no effect (ledger/ledger#2061)

- Fix unrounding for equity

- Fix SIGABRT when python subcommand raises an exception

- Improve XML reports

- Support building on older versions of CMAKE (less than 3.7)

- Fix compilation with Boost 1.76 (ledger/ledger#2030)

- Fix Msys2 MinGW build (ledger/ledger#1905)

- Fix unicode problems on Windows (ledger/ledger#1986)

- Fix the issue that with Boost >= 1.77 `include` directive cannot find the file
  to include for stdin (`-f -`). Also for `-f -` when `include` cannot find the
  file it reports the error with full path now. (ledger/ledger#2057, ledger/ledger#2092)

- Fix Nix build

- Rename `quoted_rfc4180` to `quoted_rfc`, as numbers used in function names
  confuses the parser (ledger/ledger#2007).

- Numbers are no longer permitted in value expression function names.

- Various documentation improvements

## 3.2.1 (2020-05-18)

- Fix regression with expression evaluation by reverting commit
  `Correction to the way parens are parsed in query expressions` (ledger/ledger#1894)

- Fix --invert breakage by reverting commit `Change --invert to invert
  displayed amounts and totals, not amounts` (ledger/ledger#1895)

- Fix performance regression by reverting commit `Compare price
  annotations using their textual rendering` (ledger/ledger#1907)

- Fix library path issue (ledger/ledger#1885)

- Allow specifying the Python version (ledger/ledger#1893)

- Some documentation fixes

## 3.2.0 (2020-05-01)

- Port Python support to Python 3

- Entities are no longer regarded as defined due to being part of a
  cleared transaction. `--explicit` is effectively enabled by default
  and is now a no-op (PR ledger/ledger#1819)

- Add `--average-lot-prices` to show the average of lot prices

- Add support for `%F` date format specifier (ledger/ledger#1775)

- Add `commodity_price(NAME, DATE)` function

- Add `set_commodity_price(NAME, DATE)` function

- Fix buffer overflow when evaluating date

- Fix balance assertions on accounts with virtual posts (ledger/ledger#543)

- Fix segfault with `ledger print` (ledger/ledger#1850)

- Ensure that `apply` directives (like `apply account`) have the
  required argument (ledger/ledger#553)

- Format annotations using a date format that can be parsed

- Change `--invert` to invert displayed amounts and totals, not amounts
  (ledger/ledger#1803)

- Correct the way parens are parsed in query expressions

- Compare price annotations using their textual rendering

- Fix build failure with utfcpp 3.0 (ledger/ledger#1816)

- Fix build failure due to ambiguous type (ledger/ledger#1833)

## 3.1.3 (2019-03-31)

- Properly reject postings with a comment right after the flag (ledger/ledger#1753)

- Make sorting order of lot information deterministic (ledger/ledger#1747)

- Fix bug in tag value parsing (ledger/ledger#1702)

- Remove the `org` command, which was always a hack to begin with (ledger/ledger#1706)

- Provide Docker information in README

- Various small documentation improvements

## 3.1.2 (2019-02-05)

- Increase maximum length for regex from 255 to 4095 (ledger/ledger#981)

- Initialize periods from from/since clause rather than earliest
  transaction date (ledger/ledger#1159)

- Check balance assertions against the amount after the posting (ledger/ledger#1147)

- Allow balance assertions with multiple posts to same account (ledger/ledger#1187)

- Fix period duration of "every X days" and similar statements (ledger/ledger#370)

- Make option `--force-color` not require `--color` anymore (ledger/ledger#1109)

- Add `quoted_rfc4180` to allow CVS output with RFC 4180 compliant quoting.

- Add support for `--prepend-format` in accounts command

- Fix handling of edge cases in trim function (ledger/ledger#520)

- Fix auto xact posts not getting applied to account total during
  journal parse (ledger/ledger#552)

- Transfer `null_post` flags to generated postings

- Fix segfault when using `--market` with `--group-by`

- Use `amount_width` variable for budget report

- Keep pending items in budgets until the last day they apply

- Fix bug where `.total` used in value expressions breaks totals

- Make automated transactions work with assertions (ledger/ledger#1127)

- Improve parsing of date tokens (ledger/ledger#1626)

- Don't attempt to invert a value if it's already zero (ledger/ledger#1703)

- Do not parse user-specified init-file twice

- Fix parsing issue of effective dates (ledger/ledger#1722,
  [TALOS-2017-0303](https://talosintelligence.com/vulnerability_reports/TALOS-2017-0303),
  [CVE-2017-2807](https://www.cve.org/CVERecord?id=CVE-2017-2807))

- Fix use-after-free issue with deferred postings (ledger/ledger#1723,
  [TALOS-2017-0304](https://talosintelligence.com/vulnerability_reports/TALOS-2017-0304),
  [CVE-2017-2808](https://www.cve.org/CVERecord?id=CVE-2017-2808))

- Fix possible stack overflow in option parsing routine (ledger/ledger#1222,
  [CVE-2017-12481](https://www.cve.org/CVERecord?id=CVE-2017-12481))

- Fix possible stack overflow in date parsing routine (ledger/ledger#1224,
  [CVE-2017-12482](https://www.cve.org/CVERecord?id=CVE-2017-12482))

- Fix use-after-free when using `--gain` (ledger/ledger#541)

- Python: Removed double quotes from Unicode values.

- Python: Ensure that parse errors produce useful `RuntimeErrors`

- Python: Expose `journal expand_aliases`

- Python: Expose `journal_t::register_account`

- Improve bash completion

- Emacs Lisp files have been moved to https://github.com/ledger/ledger-mode

- Fix build under MSYS (32-bit).

- Fix build under Cygwin.

- Various documentation improvements

## 3.1.1 (2016-01-11)

- Added a `--no-revalued` option

- Improved Embedded Python Support

- Use `./.ledgerrc` if `~/.ledgerrc` doesn't exist

- Fixed parsing of transactions with single-character payees and comments

- Fixed crash when using `-M` with empty result

- Fixed sorting for option `--auto-match`

- Fixed treatment of `year 2015` and `Y2014` directives

- Fixed crash when using `--trace` 10 or above

- Build fix for boost 1.58, 1.59, 1.60

- Build fix for Cygwin

- Fixed Util and Math tests on Mac OS X

- Various documentation improvements

- Examples in the documentation are tested just like unit tests

- Add continuous integration (https://travis-ci.org/ledger/ledger)

## 3.1 (2014-10-05)

- Changed the definition of cost basis to preserve the original cost basis
  when a gain or loss is made (if you bought 1 AAA for $10 and then sold
  it for $12, ledger would previously take $12 as the cost; the original
  cost of $10 is preserved as the cost basis now, which addresses strange
  behavior with -B after a capital gain or loss is made).

- Incorrect automatic Equity:Capital Gains and Equity:Capital Loss entries
  are no longer generated when a commodity is sold for loss or profit.

- Support for virtual posting costs.

- The option `--permissive` now quiets balance assertions

- Removed SHA1 files due to license issues and use boost instead.

- Added option `--no-pager` to disable the pager.

- Added option `--no-aliases` to completely disable alias expansion

- Added option `--recursive-aliases` to expand aliases recursively

- Support payee `uuid` directive.

- Bug fix: when a status flag (`!` or `*`) is explicitly specified for an
  individual posting, it always has a priority over entire transaction
  status.

- Bug fix: don't lose commodity when cost is not separated by whitespace

- Improved backwards compatibility with ledger 2.x

- Build fix for GCC 4.9

- Build fix for boost 1.56

- Many improvements to ledger-mode, including fontification

- More test cases and unit tests

- Contrib: Added script to generate commodities from ISO 4217

## 3.0

Due to the magnitude of changes in 3.0, only changes that affect compatibility
with 2.x files and usage is mentioned here.  For a description of new
features, please see the manual.

- The option `-g` (`--performance`) was removed.

- The balance report now defaults to showing all relevant accounts.  This is
  the opposite of 2.x.  That is, `bal` in 3.0 does what `-s bal` did in 2.x.
  To see 2.6 behavior, use `bal -n` in 3.0.  The `-s` option no longer has any
  effect on balance reports.

## 2.6.3

- Minor fixes to allow for compilation with gcc 4.4.

## 2.6.2

- Bug fix: Command-line options, such as -O, now override init-file options
  such as -V.

- Bug fix: "cat data | ledger -f -" now works.

- Bug fix: --no-cache is now honored.  Previously, it was writing out a cache
  file named "<none>".

- Bug fix: Using %.2X in a format string now outputs 2 spaces if the state is
  cleared.

## 2.6.1

- Added the concept of "balance setting transactions":

  Setting an account's balance

  You can now manually set an account's balance to whatever you want, at
  any time.  Here's how it might look at the beginning of your Ledger
  file:

      2008/07/27 Starting fresh
          Assets:Checking      = $1,000.00
          Equity:Opening Balances

  If Assets:Checking is empty, this is no different from omitting the
  "=".  However, if Assets:Checking did have a prior balance, the amount
  of the transaction will be auto-calculated so that the final balance
  of Assets:Checking is now $1,000.00.

  Let me give an example of this.  Say you have this:

      2008/07/27 Starting fresh
          Assets:Checking          $750.00
          Equity:Opening Balances

      2008/07/27 Starting fresh
          Assets:Checking      = $1,000.00
          Equity:Adjustments

  These two entries are exactly equivalent to these two:

      2008/07/27 Starting fresh
          Assets:Checking          $750.00
          Equity:Opening Balances

      2008/07/27 Starting fresh
          Assets:Checking          $250.00
          Equity:Adjustments

  The use of the "=" sign here is that it sets the transaction's amount
  to whatever is required to satisfy the assignment.  This is the
  behavior if the transaction's amount is left empty.

  # Multiple commodities

  As far as commodities go, the = sign only works if the account
  balance's commodity matches the commodity of the amount after the
  equals sign.  However, if the account has multiple commodities, only
  the matching commodity is affected.  Here's what I mean:

      2008/07/24 Opening Balance
          Assets:Checking        = $250.00          ; we force set it
          Equity:Opening Balances

      2008/07/24 Opening Balance
          Assets:Checking      = EC 250.00          ; we force set it again
          Equity:Opening Balances

  This is an error, because $250.00 cannot be auto-balanced to match EC
  250.00.  However:

      2008/07/24 Opening Balance
          Assets:Checking        = $250.00          ; we force set it again
          Assets:Checking        EC 100.00          ; and add some EC's
          Equity:Opening Balances

      2008/07/24 Opening Balance
          Assets:Checking      = EC 250.00          ; we force set the EC's
          Equity:Opening Balances

  This is *not* an error, because the latter auto-balancing transaction
  only affects the EC 100.00 part of the account's balance; the $250.00
  part is left alone.

  Checking statement balances

  When you reconcile a statement, there are typically one or more
  transactions which result in a known balance.  Here's how you specify
  that in your Ledger data:

      2008/07/24 Opening Balance
          Assets:Checking        = $100.00
          Equity:Opening Balances

      2008/07/30 We spend money, with a known balance afterward
          Expenses:Food             $20.00
          Assets:Checking         = $80.00

      2008/07/30 Again we spend money, but this time with all the info
          Expenses:Food             $20.00
          Assets:Checking          $-20.00 = $60.00

      2008/07/30 This entry yield an 'unbalanced' error
          Expenses:Food             $20.00
          Assets:Checking          $-20.00 = $30.00

  The last entry in this set fails to balance with an unbalanced
  remainder of $-10.00.  Either the entry must be corrected, or you can
  have Ledger deal with the remainder automatically:

      2008/07/30 The fixed entry
          Expenses:Food              $20.00
          Assets:Checking           $-20.00 = $30.00
          Equity:Adjustments

  Conclusion

  This simple feature has all the utility of @check, plus auto-balancing
  to match known target balances, plus the ability to guarantee that an
  account which uses only one commodity does contain only that
  commodity.

  This feature slows down textual parsing slightly, does not affect
  speed when loading from the binary cache.

- The rest of the changes in the version is all bug fixes (around 45 of
  them).

## 2.6.0.90

- Gnucash parser is fixed.

- Fix a memory leak bug in the amount parser.

- (This feature is from 2.6, but was not documented anywhere):

  Commodities may now specify lot details, to assign in managing set
  groups of items, like buying and selling shares of stock.

  For example, let's say you buy 50 shares of AAPL at $10 a share:

      2007/01/14 Stock purchase
          Assets:Brokerage         50 AAPL @ $10
          Assets:Brokerage

  Three months later, you sell this "lot".  Based on the original
  purchase information, Ledger remembers how much each share was
  purchased for, and the date on which it was purchased.  This means
  you can sell this specific lot by price, by date, or by both.  Let's
  sell it by price, this time for $20 a share.

      2007/04/14 Stock purchase
          Assets:Brokerage         $1000.00
          Assets:Brokerage         -50 AAPL {$10} @ $20
          Income:Capital Gains     $-500.00

  Note that the Income:Capital Gains line is now required to balance
  the transaction.  Because you sold 50 AAPL at $20/share, and because
  you are selling shares that were originally valued at $10/share,
  Ledger needs to know how you will "balance" this difference.  An
  equivalent Expenses:Capital Loss would be needed if the selling
  price were less than the buying price.

  Here's the same example, this time selling by date and price:

      2007/04/14 Stock purchase
          Assets:Brokerage         $1000.00
          Assets:Brokerage         -50 AAPL {$10} [2007/01/14] @ $20
          Income:Capital Gains     $-500.00

  If you attempt to sell shares for a date you did not buy them, note
  that Ledger will not complain (as it never complains about the
  movement of commodities between accounts).  In this case, it will
  simply create a negative balance for such shares within your
  Brokerage account; it's up to you to determine whether you have them
  or not.

- To facilitate lot pricing reports, there are some new reporting
  options:

  * --lot-prices   Report commodities with different lot prices as if
                   they were different commodities.  Otherwise, Ledger
                   just gloms all the AAPL shares together.

  * --lot-dates    Separate commodities by lot date.  Every
                   transaction that uses the '@' cost specifier will
                   have an implicit lot date and lot price.

  * --lot-tags     Separate commodities by their arbitrary note tag.
                   Note tags may be specified using (note) after the
                   commodity.

  * --lots         Separate commodities using all lot information.

## 2.6

- The style for eliding long account names (for example, in the
  register report) has been changed.  Previously Ledger would elide
  the end of long names, replacing the excess length with "..".
  However, in some cases this caused the base account name to be
  missing from the report!

  What Ledger now does is that if an account name is too long, it will
  start abbreviating the first parts of the account name down to two
  letters in length.  If this results in a string that is still too
  long, the front will be elided -- not the end.  For example:

      Expenses:Cash           ; OK, not too long
      Ex:Wednesday:Cash       ; "Expenses" was abbreviated to fit
      Ex:We:Afternoon:Cash    ; "Expenses" and "Wednesday" abbreviated
      ; Expenses:Wednesday:Afternoon:Lunch:Snack:Candy:Chocolate:Cash
      ..:Af:Lu:Sn:Ca:Ch:Cash  ; Abbreviated and elided!

  As you can see, it now takes a very deep account name before any
  elision will occur, whereas in 2.x elisions were fairly common.

- In addition to the new elision change mentioned above, the style is
  also configurable:

  * --truncate leading      ; elide at the beginning
  * --truncate middle       ; elide in the middle
  * --truncate trailing     ; elide at end (Ledger 2.x's behavior)
  * --truncate abbrev       ; the new behavior

  * --abbrev-len 2          ; set length of abbreviations

  These elision styles affect all format strings which have a maximum
  width, so they will also affect the payee in a register report, for
  example.  In the case of non-account names, "abbrev" is equivalent
  to "trailing", even though it elides at the beginning for long
  account names.

- Error reporting has been greatly improving, now showing full
  contextual information for most error messages.

- Added --base reporting option, for reporting convertible commodities
  in their most basic form.  For example, if you read a timeclock file
  with Ledger, the time values are reported as hour and minutes --
  whichever is the most compact form.  But with --base, Ledger reports
  only in seconds.

  NOTE: Setting up convertible commodities is easy; here's how to use
  Ledger for tracking quantities of data, where the most compact form
  is reported (unless --base is specified):

      C 1.00 Kb = 1024 b
      C 1.00 Mb = 1024 Kb
      C 1.00 Gb = 1024 Mb
      C 1.00 Tb = 1024 Gb

- Added --ansi reporting option, which shows negative values in the
  running total column of the register report as red, using ANSI
  terminal codes; --ansi-invert makes non-negative values red (which
  makes more sense for the income and budget reports).

  The --ansi functionality is triggered by the format modifier "!",
  for example the register reports uses the following for the total
  (last) column:

      %!12.80T

  At the moment neither the balance report nor any of the other
  reports make use of the ! modifier, and so will not change color
  even if --ansi is used.  However, you can modify these report format
  strings yourself in ~/.ledgerrc if you wish to see red coloring of
  negative sums in other places.

- Added --only predicate, which occurs during transaction processing
  between --limit and --display.  Here is a summary of how the three
  supported predicates are used:

  * --limit "a>100"

      This flag limits computation to *only transactions whose amount
      is greater than 100 of a given commodity*.  It means that if you
      scan your dining expenses, for example, only individual bills
      greater than $100 would be calculated by the report.

  * --only "a>100"

      This flag happens much later than --limit, and corresponding
      more directly to what one normally expects.  If --limit isn't
      used, then ALL your dining expenses contribute to the report,
      *but only those calculated transactions whose value is greater
      than $100 are used*.  This becomes important when doing a
      monthly costs report, for example, because it makes the
      following command possible:

        ledger -M --only "a>100" reg ^Expenses:Food

      This shows only *months* whose amount is greater than 100.  If
      --limit had been used, it would have been a monthly summary of
      all individual dinner bills greater than 100 -- which is a very
      different thing.

  * --display "a>100"

      This predicate does not constrain calculation, but only display.
      Consider the same command as above:

          ledger -M --display "a>100" reg ^Expenses:Food

      This displays only lines whose amount is greater than 100, *yet
      the running total still includes amounts from all transactions*.
      This command has more particular application, such as showing
      the current month's checking register while still giving a
      correct ending balance:

          ledger --display "d>[this month]" reg Checking

    Note that these predicates can be combined.  Here is a report that
    considers only food bills whose individual cost is greater than
    $20, but shows the monthly total only if it is greater than $500.
    Finally, we only display the months of the last year, but we
    retain an accurate running total with respect to the entire ledger
    file:

          ledger -M --limit "a>20" --only "a>200" \
            --display "year == yearof([last year])" reg ^Expenses:Food

- Added new "--descend AMOUNT" and "--descend-if VALEXPR" reporting
  options.  For any reports that display valued transactions (i.e.,
  register, print, etc), you can now descend into the component
  transactions that made up any of the values you see.

  For example, say you request a --monthly expenses report:

      $ ledger --monthly register ^Expenses

  Now, in one of the reported months you see $500.00 spent on
  Expenses:Food.  You can ask Ledger to "descend" into, and show the
  component transactions of, that $500.00 by respecifying the query
  with the --descend option:

      $ ledger --monthly --descend "\$500.00" register ^Expenses

  The --descend-if option has the same effect, but takes a value
  expression which is evaluated as a boolean to locate the desired
  reported transaction.

- Added a "dump" command for creating binary files, which load much
  faster than their textual originals.  For example:

      ledger -f huge.dat -o huge.cache dump
      ledger -f huge.cache bal

  The second command will load significantly faster (usually about six
  times on my machine).

- There have a few changes to value expression syntax.  The most
  significant incompatibilities being:

  * Equality is now ==, not =
  * The U, A, and S functions now requires parens around the argument.
    Whereas before At was acceptable, now it must be specified as
    A(t).
  * The P function now always requires two arguments.  The old
    one-argument version P(x) is now the same as P(x,m).

  The following value expression features are new:

  * A C-like comma operator is supported, where all but the last term
    are ignored.  The is significant for the next feature:
  * Function definitions are now supported.  Scoping is governed
    by parentheses.  For example:
      (x=100, x+10)      ; yields 110 as the result
      (f(x)=x*2,f(100))  ; yields 200 as the result
  * Identifier names may be any length.  Along with this support comes
    alternate, longer names for all of the current one-letter value
    expression variables:

     Old    New
     ---    ---
     m      now
     a      amount
     a      amount
     b      cost
     i      price
     d      date
     X      cleared
     Y      pending
     R      real
     L      actual
     n      index
     N      count
     l      depth
     O      total
     B      cost_total
     I      price_total
     v      market
     V      market_total
     g      gain
     G      gain_total
     U(x)   abs(x)
     S(x)   quant(x), quantity(x)
            comm(x), commodity(x)
            setcomm(x,y), set_commodity(x,y)
     A(x)   mean(x), avg(x), average(x)
     P(x,y) val(x,y), value(x,y)
            min(x,y)
            max(x,y)

- There are new "parse" and "expr" commands, whose argument is a
  single value expression.  Ledger will simply print out the result of
  evaluating it.  "parse" happens before parsing your ledger file,
  while "expr" happens afterward.  Although "expr" is slower as a
  result, any commodities you use will be formatted based on patterns
  of usage seen in your ledger file.

  These commands can be used to test value expressions, or for doing
  calculation of commoditized amounts from a script.

  A new "--debug" will also dump the resulting parse tree, useful for
  submitting bug reports.

- Added new min(x,y) and max(x,y) value expression functions.

- Value expression function may now be defined within your ledger file
  (or initialization file) using the following syntax:

      @def foo(x)=x*1000

  This line makes the function "foo" available to all subsequent value
  expressions, to all command-line options taking a value expression,
  and to the new "expr" command (see above).

## 2.5

- Added a new value expression regexp command:
    C//  compare against a transaction amount's commodity symbol

- Added a new "csv" command, for outputting results in CSV format.

- Ledger now expands ~ in file pathnames specified in environment
  variables, initialization files and journal files.

- Effective dates may now be specified for entries:

      2004/10/03=2004/09/30 Credit card company
          Liabilities:MasterCard         $100.00
          Assets:Checking

  This entry says that although the actual transactions occurred on
  October 3rd, their effective date was September 30th.  This is
  especially useful for budgeting, in case you want the transactions
  to show up in September instead of October.

  To report using effective dates, use the --effective option.

- Actual and effective dates may now be specified for individual
  transactions:

      2004/10/03=2004/09/30 Credit card company
          Liabilities:MasterCard         $100.00
          Assets:Checking                         ; [2004/10/10=2004/09/15]

  This states that although the actual date of the entry is
  2004/10/03, and the effective date of the entry is 2004/09/30, the
  actual date of the Checking transaction itself is 2004/10/10, and
  its effective date is 2004/09/15.  The effective date is optional
  (just specifying the actual date would have read "[2004/10/10]").

  If no effective date is given for a transaction, the effective date
  of the entry is assumed.  If no actual date is given, the actual
  date of the entry is assumed.  The syntax of the latter is simply
  [=2004/09/15].

- To support the above, there is a new formatting option: "%d".  This
  outputs only the date (like "%D") if there is no effective date, but
  outputs "ADATE=EDATE" if there is one.  The "print" report now uses
  this.

- To support the above, the register report may now split up entries
  whose component transactions have different dates.  For example,
  given the following entry:

      2005/10/15=2005/09/01 iTunes
          Expenses:Music                 $1.08 ; [2005/10/20=2005/08/01]
          Liabilities:MasterCard

  The command "ledger register" on this data file reports:

      2005/10/20 iTunes   Expenses:Music            $1.08    $1.08
      2005/10/15 iTunes   Liabilities:MasterCard   $-1.08        0

  While the command "ledger --effective register" reports:

      2005/08/01 iTunes   Expenses:Music            $1.08    $1.08
      2005/09/01 iTunes   Liabilities:MasterCard   $-1.08        0

  Although it appears as though two entries are being reported, both
  transactions belong to the same entry.

- Individual transactions may now be cleared separately.  The old
  syntax, which is still supported, clears all transactions in an
  entry:

      2004/05/27 * Book Store
          Expenses:Dining                 $20.00
          Liabilities:MasterCard

  The new syntax allows clearing of just the MasterCard transaction:

      2004/05/27 Book Store
          Expenses:Dining                 $20.00
          * Liabilities:MasterCard

  NOTE: This changes the output format of both the "emacs" and "xml"
  reports.  ledger.el uses the new syntax unless the Lisp variable
  `ledger-clear-whole-entries' is set to t.

- Removed Python integration support.

- Did much internal restructuring to allow the use of libledger.so in
  non-command-line environments (such as GUI tools).

## 2.4.1

- Corrected an issue that had inadvertently disabled Gnucash support.

## 2.4

- Both `-$100.00` and `$-100.00` are now equivalent amounts.

- Simple, inline math (using the operators +-/*, and/or parentheses)
  is supported in transactions.  For example:

      2004/05/27 Book Store
          Expenses:Dining                 $20.00 + $2.50
          Liabilities:MasterCard

  This won't register the tax/tip in its own account, but might make
  later reading of the ledger file easier.

- Use of a "catch all" account is now possible, which auto-balances
  entries that contain _only one transaction_.  For sanity's sake this
  is not used to balance all entries, as that would make locating
  unbalanced entries a nightmare.  Example:

      A Liabilities:MasterCard

      2004/05/27 Book Store
          Expenses:Dining                 $20.00 + $2.50

  This is equivalent to the entry in the previous bullet.

- Entries that contain a single transaction with no amount now always
  balance, even if multiple commodities are involved.  This means that
  the following is now supported, which wasn't previously:

      2004/06/21 Adjustment
          Retirement          100 FUNDA
          Retirement          200 FUNDB
          Retirement          300 FUNDC
          Equity:Adjustments

- Fixed several bugs relating to QIF parsing, budgeting and
  forecasting.

- The configure process now looks for libexpat in addition to
  searching for libxmlparse+libxmltok (how expat used to be packaged).

## 2.3

- The directive "!alias ALIAS = ACCOUNT" makes it possible to use
  "ALIAS" as an alternative name for ACCOUNT in a textual ledger file.
  You might use this to associate the single word "Bank" with the
  checking account you use most, for example.

- The --version page shows the optional modules ledger was built with.

- Fixed several minor problems, plus a few major ones dealing with
  imprecise date parsing.

## 2.2

- Ledger now compiles under gcc 2.95.

- Fixed several core engine bugs, and problems with Ledger's XML data
  format.

- Errors in XML or Gnucash data now report the correct line number for
  the error, instead of always showing line 1.

- 'configure' has been changed to always use a combination of both
  compile and link tests for every feature, in order to identify
  environment problems right away.

- The "D <COMM>" command, released in 2.1, now requires a commoditized
  amount, such as "D $1,000.00".  This sets not only the default
  commodity, but several flags to be used with all such commodities
  (such as whether numbering should be American or European by
  default).  This entry may be used be many times; the most recent
  seen specifies the default for entries that follow.

- The binary cache now remembers the price history database that was
  used, so that if LEDGER_PRICE_DB is silently changed, the cache will
  be thrown away and rebuilt.

- OFX data importing is now supported, using libofx
  (http://libofx.sourceforge.net).  configure will check if the
  library is available.  You may need to add CPPFLAGS or LDFLAGS to
  the command-line for the appropriate headers and library to be
  found.  This support is preliminary, and as such is not documented
  yet.

- All journal entries now remember where they were read from.  New
  format codes to access this information are: %S for source path, %B
  for beginning character position, and %E for ending character
  position.

- Added "pricesdb" command, which is identical to "prices" except that
  it uses the same format as Ledger's usual price history database.

- Added "output FILE" command, which attempts to reproduce the input
  journal FILE exactly.  Meant for future GUI usage.  This command
  relies on --write-hdr-format and --write-xact-format, instead of
  --print-format.

- Added "--reconcile BALANCE" option, which attempts to reconcile all
  matching transactions to the given BALANCE, outputting those that
  would need to be "cleared" to match it.  Using by the
  auto-reconciling feature of ledger.el (see below).

  "--reconcile-date DATE" ignores any uncleared transactions after
  DATE in the reconciling algorithm.  Since the algorithm is O(n^2)
  (where 'n' is the number of uncleared transactions to consider),
  this could have a substantial impact.

- In ledger.el's *Reconcile* mode ('C-c C-r' from a ledger-mode file):

  * 'a' adds a missing transaction
  * 'd' deletes the current transaction
  * 'r' attempts to auto-reconcile (same as 'C-u C-c C-r')
  * 's' or 'C-x C-s' will save the ledger data file and show the
    currently cleared balance
  * 'C-c C-c' commits the pending transactions, marking them cleared.

  This feature now works with Emacs 21.3.
  Also, the reconciler no longer needs to ask "how far back" to go.

- To support the reconciler, textual entries may now have a "!" flag
  (pending) after the date, instead of a "*" flag (cleared).

- There are a new set of value expression regexp commands:
  * c//  entry code
  * p//  payee
  * w//  short account name
  * W//  full account name
  * e//  transaction note

  This makes it possible to display transactions whose comment field
  matches a particular text string.  For example:

      ledger -l e/{tax}/ reg

  prints out all the transactions with the comment "{tax}", which
  might be used to identify items related to a tax report.

## 2.1

- Improved the autoconf system to be smarter about finding XML libs

- Added --no-cache option, to always ignore any binary cache file

- `ledger-reconcile' (in ledger.el) no longer asks for a number of days

- Fixed %.XY format, where X is shorter than the string generated by Y

- New directive for text files: "D <COMM>" specifies the default commodity
  used by the entry command

## 2.0

This version represents a full rewrite, while preserving much of the
original data format and command-line syntax.  There are too many new
features to describe in full, but a quick list: value expressions,
complex date masks, binary caching of ledger data, several new
reporting options, a simple way to specify payee regexps, calculation
and display predicates, and two-way Python integration.  Ledger also
uses autoconf now, and builds as a library in addition to a
command-line driver.

### Differences from 1.7

- changes in option syntax:

  -d now specifies the display predicate.  To give a date mask similar
  to 1.7, use the -p (period) option.

  -P now generates the "by payee" report.  To specify a price database
  to use, use --price-db.

  -G now generates a net gain report.  To print totals in a format
  consumable by gnuplot, use -J.

  -l now specifies the calculation predicate.  To emulate the old
  usage of "-l \$100", use: -d "AT>100".

  -N is gone.  Instead of "-N REGEX", use: -d "/REGEX/?T>0:T".

  -F now specifies the report format string.  The old meaning of -F
  now has little use.

  -S now takes a value expression as the sorting criterion.  To get
  the old meaning of "-S", use "-S d".

  -n now means "collapse entries in the register report".  The get the
  old meaning of -n in the balance report, use "-T a".

  -p now specifies the reporting period.  You can convert commodities
  in a report using value expressions.  For example, to display hours
  at $10 per hour:

      -T "O>={0.01h}?{\$10.00}*O:O"

  Or, to reduce totals, so that every $417 becomes 1.0 AU:

      -T "O>={\$0.01}?{1.0 AU}*(O/{\$417}):O"

- The use of "+" and "-" in ledger files to specify permanent regexps
  has been removed.

- The "-from" argument is no longer used by the "entry" command.
  Simply remove it.

### Features new to 2.0

- The most significant feature to be added is "value expressions".
  They are used in many places to indicate what to display, sorting
  order, how to calculate totals, etc.  Logic and math operators are
  supported, as well as simple functions.  See the manual.

- If the environment variable LEDGER_FILE (or LEDGER) is used, a
  binary cache of that ledger is kept in ~/.ledger-cache (or the file
  given by LEDGER_CACHE).  This greatly speeds up subsequent queries.
  Happens only if "-f" or "--file" is not used.

- New 'xml' report outputs an XML version of what "register" would
  have displayed.  This can be used to manipulate reported data in a
  more scriptable way.

  Ledger can also read as input the output from the "xml" report.  If
  the "xml" report did not contain balanced entries, they will be
  balanced by the "<Unknown>" account.  For example:

      ledger reg rent

  displays the same results as:

      ledger xml rent | ledger -f - reg rent

- Regexps given directly after the command name now apply only to
  account names.  To match on a payee, use "--" to separate the two
  kinds of regexps.  For example, to find a payee named "John" within
  all Expenses accounts, use:

      ledger register expenses -- john

  Note: This command is identical (and internally converted) to:

      ledger -l "/expenses/|//john/" register

- To include entries from another file into a specific account, use:

      !account ACCOUNT
      !include FILE
      !end

- Register reports now show only matching account transactions.  Use
  "-r" to see "related accounts" -- the account the transfer came from
  or went to (This was the old behavior in 1.x, but led to confusion).
  "-r" also works with balance reports, where it will total all the
  transactions related to your query.

- Automated transactions now use value expressions for the predicate.
  The new syntax is:

      = VALUE-EXPR
        TRANSACTIONS...

  Only one VALUE-EXPR is supported (compared to multiple account
  regexps before).  However, since value expression allow for logic
  chaining, there is no loss of functionality.  Matching can also be
  much more comprehensive.

- If Boost.Python is installed (libboost_python.a), ledger can support
  two-way Python integration.  This feature is enabled by passing
  --enable-python to the "configure" script before building.  Ledger
  can then be used as a module (ledger.so), as well as supporting
  Python function calls directly from value expressions.  See main.py
  for an example of driving Ledger from Python.  It implements nearly
  all the functionality of the C++ driver, main.cc.

  (This feature has yet to mature, and so is being offered as a beta
  feature in this release.  It is mostly functional, and those curious
  are welcome to play with it.)

- New reporting options:

  * "-o FILE" outputs data to FILE.  If "-", output goes to stdout (the
    default).

  * -O shows base commodity values (this is the old behavior)
  * -B shows basis cost of commodities
  * -V shows market value of commodities
  * -g reports gain/loss performance of each register item
  * -G reports net gain/loss over time
  * -A reports average transaction value (arithmetic mean)
  * -D reports each transaction's deviation from the average

  * -w uses 132 columns for the register report, rather than 80.  Set
   the environment variable LEDGER_WIDE for this to be the default.

  * "-p INTERVAL" allows for more flexible period reporting, such as:

      monthly
      every week
      every 3 quarters
      weekly from 12/20
      monthly in 2003
      weekly from last month until dec

  * "-y DATEFMT" changes the date format used in all reports.  The
    default is "%Y/%m/%d".

    -Y and -W print yearly and weekly subtotals, just as -M prints
    monthly subtotals.

  * --dow shows cumulative totals for each day of the week.

  * -P reports transactions grouped by payee

  * -x reports the payee as the commodity; useful in some cases

  * -j and -J replace the previous -G (gnuplot) option.  -j reports the
    amounts column in a way gnuplot can consume, and -J the totals
    column.  An example is in "scripts/report".

  * "--period-sort EXPR" sorts transactions within a reporting period.
    The regular -S option sorts all reported transactions.

## 1.7

- Pricing histories are now supported, so that ledger remembers the
  historical prices of all commodities, and can present register
  reports based on past and present market values as well as original
  cost basis.  See the manual for more details on the new option
  switches.

## 1.6

- Ledger can now parse timeclock files.  These are simple timelogs
  that track in/out events, which can be maintained using my timeclock
  tool.  By allowing ledger to parse these, it means that reporting
  can be done on them in the same way as ledger files (the commodity
  used is "h", for hours); it means that doing things like tracking
  billable hours for clients, and invoicing those clients to transfer
  hours into dollar values via a receivable account, is now trivial.
  See the docs for more on how to do this.

- Began keeping a NEWS file. :)
