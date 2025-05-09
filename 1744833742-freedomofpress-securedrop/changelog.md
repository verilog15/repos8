# Changelog

## 2.13.0~rc1

## 2.12.1

Note: this is an Admin Workstation-only release. Servers will not receive an update.

### Ubuntu 24.04 (Noble) upgrade

* Make noble-migration playbook smarter for SSH-over-Tor (#7484)
* Extend reboot timeout to 600 seconds (#7484)

## 2.12.0

### Ubuntu 24.04 (Noble) upgrade

* Add CI staging support for Noble (#7360)
* Add the `sdssh` group before using it in ACLs (#7426)
* Set a fixed machine-id to ensure phased Noble updates are consistent (#7424)
* Add script to upgrade from Ubuntu 20.04 (Focal) to Noble (#7406)
* Add support for manual upgrades from Focal to Noble (#7427)
* Correct end-of-life date for Ubuntu 20.04 (Focal) (#7459, #7465)
* Remove `/etc/apt/sources.list.d/original.list` (#7462, #7467)
* Remove `systemd-resolved` and explicitly install `systemd-hwe-hwdb` (#7466,
  #7472)
* Ignore RUSTSEC-2025-0014 (#7470, #7471)

### Web applications and APIs

* Update date string formatting to follow ISO8061 standards (#6413)
* Update `redwood` to used stabilized `File::create_new()` (#7405)
* Update Rust toolchain to 1.84.1 (#7423, #7437)
* Add support for disabling previously-supported languages (#7443)
* Remove Hindi as a supported language (#7446)
* Add support for disabling previously-supported languages (#7443, #7451)
* Ignore Safety alerts:
  * Ignore Safety 73969 in `jinja2` (#7402)
* Update dependencies:
* Update Rust `openssl` dependency to  0.10.70 (#7435)

### Operations

* Improve `securedrop-admin` error messaging (#7272)
* Update `systemd` services using `Type=exec` to use `Type=simple` (#7404)
* Add a single script to manage Redis authentication changes (#7409)
* Ensure `/etc/iptables` exists before writing to it (#7417)
* Fix `systemd` `ConditionPathExists` syntax (#7434)
* Ignore Safety alerts:
  * Ignore Safety 74221, 74261 in `ansible-core` (#7402)

### Development

* Update `backport.py` utility script (#7375)
* Remove unused translator credits file (#7397)
* Add support for Podman in `make dev-tor` (#7163)
* Update `make update-python3-requirements` to use a container (#7400)
* Add Github Actions workflow linting via `zizmor` (#7401)
* Add `flake8-bugbear` rules to `ruff` config (#7403)
* Update `testinfra` tests to resolve dpkg lock contention (#7418)
* Update Tor Browser tests to be parameterized by locale (#7414)
* Update `testinfra` tests to handle unapplied phased updates (#7420)
* Update `testinfra` tests to speed up `pam_ecryptfs` check (#7428)
* Update dependency review documentation to reflect Rust and Python differences (#7436)
* Add `libvirt-prod-noble` molecule scenario (#7449)
* Update dependencies:
  * Update `pip` to 25.0 (#7429)

## 2.11.1

* Modify the `securedrop-noble-migration-check` program to avoid triggering
  spurious OSSEC alerts (#7394)

## 2.11.0

The main focus for this release was to prepare SecureDrop servers for upgrading
to Ubuntu 24.04 (Noble) next year. Other maintenance changes are also included.

### Ubuntu 24.04 (Noble) upgrade

* Support building packages on noble (#7273, #7247, #7319)
* Add a noble migration check script (#7334, #7363, #7369, #7378)
* Use `Type=exec` instead of `Type=oneshot` for systemd units (#7350)
* Make Ansible variables distro-agnostic (#7356)
* Apply `grsec_lock` once only (#7353)
* Stop setting `vm.heap_stack_gap` and `net.ipv4 sysctl` flags via Ansible (#7324)
* Use `sdssh` group instead of internal-only `ssh` group for access control (#7317, #7355)
* Add timed job to clean out old OSSEC diff and state files (#7327)
* Remove ufw from new and existing installs (#7315, #7377)
* Update apache config templates to be distro-agnostic (#7301)
* Install backup script on app server via Debian package (#7331)
* Ensure `sources.list` is absent on noble (#7342)
* Overwrite `sources.list.d/ubuntu.sources` on noble (#7307)

### Web applications

* Add a banner in the Journalist Interface, in preparation for the noble migration (#7348)
* Use `sqlalchemy.LargeBinary` instead of deprecated `Binary` (#7264)
* Upgrade sequoia-openpgp from 1.21.1 to 1.21.2 (#7248)
* Import escape from markupsafe, not flask (#7252)
* Update UI strings based on translator feedback (#7370)
* Ignore safety alerts:
  * ignore Safety 73711 in cryptography (#7339)
  * ignore Safety 73889, 73969 in werkzeug (#7361)

### Operations

* Regenerate Redis password on restoring from server backup (#7328)
* Replace reboot-flag cron job with a systemd timer (#7337)
* Remove haveged package, if installed (#7335, 7341)
* Don't install apt-transport-https transitional package (#7303)
* Remove unused Ansible `restrict_direct_access_{app,mon}` roles (#7302)
* Remove unused Ansible `sysctl_flags_ipv6 variables` (#7300)
* Prompt `sdadmin` for the default SSH username (#7309)
* Remove unused `load_iptables` script (#7282)
* Remove unused SSHd config from cloud-init (#7318)
* Remove stray Ubuntu file `/etc/apt/apt.conf.d/zzzz-temp-installer-unattended-upgrade` if it exists (#7380)

### Development and CI

* Publish versions of packages with debug symbols (#7347, #7365)
* Preserve screenshots from translation test CI job (#7240)
* Make `backport.py` more flexible for complex pull requests (#7260)
* Install xz-utils in diffoscope CI job (#7344)
* Don't return `True` from `test_swap_disabled` for monitor server, skip test instead (#7320)
* Run admin CI tests on bookworm (#7212)
* Use a single pass in ansible to install local packages (#7261)
* Upgrade tbselenium from 0.8.1 to 0.9.0 (#7274, #7271)
* Update geckodriver from 0.33.0 to 0.35.0 (#7268)
* Standardize git message formats in version updater (#7263)
* Speed up `update-python3-dependencies` Makefile target using uv (#7234)
* Upgrade ruff, remove black, add ruff formatting fixes (#7233, #7246)
* Remove unused `devops/scripts/aws-jenkins-venv.sh` (#7238)
* Ignore safety alerts:
  * Ignore CVE-2024-8775 in ansible-core (#7269)
* Update dependencies:
  * Upgrade cargo-vet from 0.9.0 to 0.10.0 (#7343)
  * Upgrade Rust toolchain from 1.78.0 to 1.81.0 (#7232)

#### In support of Ubuntu 24.04 (Noble) upgrade

* Support noble dev environment (#7249)
* Run basic lint CI against Ubuntu noble and Python 3.12 (#7242)
* Remove tests checking that no apparmor profiles are complaining (#7308)
* Remove `test_securedrop_application_apt_dependencies` test (#7305)
* Inspect `grsec_lock` as root in testinfra (#7304)
* Upgrade paramiko from 2.7.2 to 2.10.6 (#7280, #7321)

## 2.10.1

* Update translations (#7143, #7259)

## 2.10.0

This release contains fixes for issues described in the most recent security audit by 7A Security, see
our [blog post](https://securedrop.org/news/securedrop-2_10_0-released/) for more details. It also contains other maintenance fixes.

### Security

* Don't allow admins to look up arbitrary users' TOTP secrets via the web (SEC-01-001 WP4)
* Validate user provided same password back to server (SEC-01-002 WP4)
* Require POST requests for `/logout` for CSRF protection (SEC-01-003 WP4)
* Set password for redis access (SEC-01-008 WP3)
* Set `SameSite=Strict` on all cookies for more CSRF protection

### Web applications
* Dependency updates:
  * sequoia-openpgp (Rust crate) from 1.20.0 to 1.21.1 (#7197)
  * setuptools from 56.0.0 to 70.3.0 for CVE-2024-6345 (#7205, #7214)
  * openssl (Rust crate) from 0.10.60 to 0.10.66 for RUSTSEC-2024-0357 (#7206)

### Journalist Workstation
* Dependency updates:
  * setuptools from 56.0.0 to 70.3.0 for CVE-2024-6345 (#7205, #7214)
  * Remove d2to1 and pbr (#7205)

### Development
* Don't point people to the decommissioned SecureDrop forum (#7204)
* Migrate all CI jobs to GitHub Actions (#7216, #7217, #7218, #7219, #7220, #7222, #7223)
* Improve staging job by using upstream gcloud-sdk image and enforcing GCE VM lifespan (#7215, #7224)
* Dependency updates:
  * certifi from 2023.7.22 to 2024.7.4 for CVE-2024-39689 (#7199)
  * Remove pytest-catchlog (#7199)

## 2.9.0

### Web applications
* Added accessibility improvements (#6536)
* Dependency updates:
  * (Rust) sequoia-openpgp from 1.17.0 to 1.20.0 (#7178)

### Journalist API
* Added support for partial content requests for submissions and replies endpoints (#7160, #7189)

### Operations
* Added opt-in Tor PoW defense for the Source Interface (#7175)
* Updated SecureDrop signing key with new expiry date of 2027-05-24 (#7167)

### Development
* Updated rust toolchain to version 1.78.0 (#7147)
* Added random file generation in loaddata.py (#7161)
* Fixed loaddata.py date generation bug (#7156)
* Updated test signing key (#7150)
* Added persistence for onion addresses created with `make dev-tor` (#7124)
* Ignored safety alerts 66278, 65212, 65401, 65193, 65510, 65511, 66700, 66777, 66704, 66710, 70612, 71064, 42926, 51385, 53048, 53868, 53869, 54219, 54421, 59473, 60026, 65505, 66667, 71591, 71594, 71595, 71608, 71680, 71681, 71684, 70895 (#7145, #7149, #7157, #7170, #7179, #7190)
* Added support for development virtualenv in Debian 12 (#7154)
* Dependency updates:
  * black from 22.3.0 to 24.3.0 (#7144)
  * pillow from 10.2.0 to 10.3.0 (#7149)

## 2.8.0

### Web applications
* Updated strings based on translator feedback (#7057)
* Improved redwood stream performance and testing (#7070)
* Updated wordlist to remove potentially confusing or offensive terms (#7024)
* Dependency changes:
  * cryptography from 41.0.3 to 41.0.7 (#7086)
  * jinja2 from 3.0.2 to 3.1.3 (#7107)
  * is-terminal rust crate from 0.4.9 to 0.4.12 (#7114)
  * openssl rust crate from 0.10.57 to 0.10.60 (#7083)

### Operations
* Updated copyright strings to reference 2024 (#7099)
* Removed obsolete mitigation for CVE-2019-3462 (#7053)
* Improved logic for installing admin tool apt dependencies in Tails (#7088)
* Added support for Tails 6 to admin tools (#7116)
* Updated GUI updater to use wayland QT plugin by default, falling back to xcb (#7134)
* Dependency changes:
  * Ansible from 6.7.0 to 8.7.0 (#7116)
  * cffi from 1.14.5 to 1.16.0 (#7116)
  * pyyaml from 5.4.1 to 6.0.1 (#7116)

### CI
* Updated CI to verify that the demo container builds and runs (#7052)
* Updated GCE CI machine type to c2-standard-8 (#7087)
* Moved various CI jobs to Github Actions (#6969)
* Fixed cargo-vet binary caching (#7065)
* Upgraded to cargo-vet 0.9.0 (#7101)
* Enabled dependabot for Github Actions (#7102)
* Fixed broken apt caches in staging-test-with-rebase job (#7110)
* Dependabot updates (#7105, #7104, #7108)

### Development
* Updated packaging logic to exclude config.py (#7014)
* Fixed broken link in contributing.md (#7028)
* Added option to specify git remote for backport script (#7044)
* Updated functional tests to run under Selenium 4 (#7100)
* Updated docker run parameters to only pass -it if a tty is available (#7098)
* Updated rust toolchain in CI and Dockerfiles to 1.74.1 (#7091)
* Decreased cargo audit error threshold (#7083)
* Fixed hot reload functionality in dev environment (#7120)
* Dependency changes:
  * MarkupSafe from 2.0.2 to 2.1.2 (#7006)
  * Selenium from 3.141.0 to 4.16.0 (#7100)
  * tbselenium from 0.5.2 to 0.8.1 (#7100)
  * jinja2 from 3.0.2 to 3.1.3 (#7109)
  * peewee from 3.15.0 to 3.17.1 (#7112)
  * diffoscope from 236 to 256 (#7125)
  * pillow from 10.0.1 to 10.2.0 (#7107)
  * semgrep from 0.98.0 to 1.57.0 (#7107)
* Updated ignored safety alerts: 61893, 62019, 63066, 63227, 65647 (#7085, #7100, #7122)

## 2.7.0

### Web applications
* Use Sequoia and redwood instead of gnupg and pretty_bad_protocol for GPG operations (#6891, #6884, #6913, #6912, #6925, #6926, #6949, #6958, #6892, #6948, #6946, #6970, #6975, #6972, #6983, #6981, #6998, #7000, #7026, #7029, #7035, #7023, #7071)
* Update translation workflow to support continuous updates (#6953, #6954, #6985, #6997, #6984, #7010, #7034)
* Update French diceware wordlist (#6936)
* Replace pretty-bad-protocol dependency with vendored version (#6836, #6907)
* Import Markup and escape from markupsafe (#6964)
* Update wordlist to remove potentially confusing or offensive terms (#7021)
* Validate the submission key,disable Journalist and Source Interfaces if a weak key is found (#7059)
* Dependency changes:
  * Update cryptography from 41.0.1 to 41.0.3 (#6940)
  * Upgrade sequioa-openpgpg from 1.16.1 to 1.17.0 (#7041)

### Operations

* Remove Ansible check for installed Tor version on servers (#6894)
* Miscellaneous demo server fixes (#6935, #6994)
* Update default Dockerfile application versions:
  * geckodriver to 0.33.0 (#6957)
  * Firefox to 115esr, Tor Browser to 13.0 (#7001)
* Update securedrop-admin tooling to reject weak GPG keys with a SHA-1 signature (#6928)
* Use systemd timers to check for disconnected submissions and source listings (#7009)
* Dependency changes:
  * Update Ansible from 2.9.26 to 6.7.0 (ansible-core version 2.13.7) (#6830)

### CI

* Replace bandit, flake8, pylint, and isort with ruff; added more checks (#6885, #6932, #6961, #6995)
* Update CI to run all non-staging tests on i18n branches (#6923)
* Reduce mypy execution time by skipping redwood compilation and parsing stub (#6971)
* Miscellaneous CI updates (#6844, #6920)

### Development

* Improve printing of apparmor denials in testinfra suite (#6883)
* Set "ia" as unavailable locale, "eo" as test locale  (#6919)
* Add script to auto-backport PRs to release branches (#6875)
* Miscellaneous development updates (#6842, #6865, #6871, #6882)
* Update build script to record commit details (#7019, #7038)
* Dependency changes:
  * Remove boto and boto3 dependencies (#6890)
  * Remove hypothesis dependency (#6893)
  * Update certifi from 2022.12.7 to 2023.7.22 (#6900)
  * Update pillow from 9.3.0 to 10.0.1 (#6959)
  * Update markupsafe from 2.0.1 to 2.1.2 (#7006)
* Miscellaneous changes (#7008)

## 2.6.1

### Operations

* Fixed bug preventing Journalist Workstation provisioning. (#6905)

## 2.6.0

### Web applications
* Don't treat Tor Browser for Android as desktop Tor Browser (#6573)
* Hash journalist passphrases using the argon2id algorithm (#6655, #6657)
* Remove use of global i18n variables (#6420, #6681)
* Explicitly retrieve SDConfig and make it immutable (#5761, #6563)
* Remove "LOGIN_HARDENING" global and reduce SECUREDROP_ENV usage (#3600, #6585)
* Officially only support SQLite as the database backend (#6700, #6707)
* Add `<title>` blocks for better accessibility (#6313, #6738)
* Set `Cross-Origin-Resource-Policy: same-origin` (#6768)
* Automatically and regularly remove pending sources (#6488, #6785)
* Fixed display bug when duplicate locales are specified (#6853)
* Dependency changes:
  * Update cryptography from 39.0.1 to 41.0.1 (#6855)
  * Update mod-wsgi from 4.6.7 to 4.9.4 (#6775)
  * Update pytz from 2017.3 to 2022.2.1 (#6569, #6571)
  * Update pycparser from 2.20 to 2.21 (#6618, #6686)
  * Update redis from 3.5.3 to 4.5.4 (#6783, #6867)
  * Update requests from 2.26.0 to 2.31.0 (#6821)
  * Update wheel from 0.33.6 to 0.38.4 (#6680)
  * Remove passlib (#6631)
  * Remove pyotp (#5613, #6679)

### Journalist Workstation
* Add a GNOME shell extension (#6531, #6712, 6847)

### Operations
* Update SecureDrop release key expiry date to 2024-07-08 (#6803, #6804, #6819)
* Switch cron jobs to systemd timers (#6705, #6748, #6780)
* Have unattended-upgrades automatically remove unused dependencies (#6762, #6791)
* Drop securedrop-grsec metapackage, moved to kernel-builder (#6328, #6553)
* Disable Ubuntu Pro's ua-timer and esm-cache jobs (#6773, #6781)
* Cleanup old Python 3.5 references (#6586)
* Cleanup manual `sys.path` additions (#6589)
* Reorganize Debian packaging, having debhelper do most of the work and other cleanup (#6544)
* Reorganize OSSEC packaging to use debhelper and other cleanup (#6754)
* Use plain container for package building, not molecule (#6706, #6754)

### Development
* Remove "sh" dependency (#6547, #6562, #6580)
* Updated development dependencies:
  * mypy from 0.761 to 1.0.0 (#6578, #6744)
  * pytest to 7.2.0 and pytest-xdist to 3.0.2 (#6689)
  * pillow from 9.0.1 to 9.3.0 (#6689)
  * shellcheck to 0.9.0, using shellcheck-py (#6715, #6719)
* Test improvements:
  * Create explicit fixture for rqworker (#6551)
  * Run flake8 for alembic migrations (#6576, #6590)
  * Enable yamllint's strict mode (#6575, #6622)
* Test journalist GUI on Python 3.9 + bullseye for Tails 5 (#6635, #6645, )
* Use a slimmer container for `make dev` (#6620, #6625)
* Switch SELinux contexts when running containers (#6666)
* Add "safe.directory" setting for dev environment on newer Git versions (#6674, #6682)
* Make developer container compatible with Apple Silicon (#6478, #6675, #6692)
* Add `make otp` helper to print or copy dev OTP code (#6621, #6673)
* Configure git hooks when virtualenv is created (#6683)
* Fix a number of spelling issues across the entire repository (#6670)
* Have black and isort lint everything, configure via pyproject.toml (#6722)
* Print SecureDrop Workstation `config.json` when using `make dev-tor` (#6736)
* Verify all packages except `securedrop-app-code` are reproducible in CI (#6754)
* Check for accessibility issues during pageslayout tests (#6745)
* Re-sign test key so it doesn't have SHA-1 signatures (#6797)

## 2.5.2

### Web applications
- Updated dependencies:
  - werkzeug from 2.0.2 to 2.2.3, markupsafe from 2.0.1 to 2.1.2, Flask-SQLAlchemy from 2.4.0
 to 2.5.1 (#6752)
  - cryptography from 3.4.7 to 39.0.1 (#6746, #6751)
  - certifi from 2017.7.27.1 to 2022-12.07 (#6708)
  - flask from 2.0.2 to 2.0.3 (#6624, #6634)

### Development
- Fixed staging app creation in source.wsgi, added application smoke test in testinfra (#6742)
- Updated safety ignore list, adding alerts: 51668, 52322, 52495, 52510, 52518 (#6708, #6731)
- Updated expected file ownership variables in testinfra configuration (#6711)

## 2.5.1

### Security

* Users and permissions: /var/www/securedrop is now root-owned, but world-readable. Python code, including manage.py, is all executed as the www-data user.

## 2.5.0

### Web Applications

* Added unified Redis-based session handling for the Journalist Interface and API (#6403, #6584)
* Added supported languages list and updated i18n tooling to process all languages available in Weblate (#6557, #6566, #6614)
* Fixed double character escaping of organisation names in the Source and Journalist Interface (#6550)
* Removed SASS from build process, added pure CSS for both web applications (#6529)
* Updated string referencing submissions to use consistent terminology (#6543)
* Updated application dependencies: mako from 1.07 to 1.2.2 (#6535)
* Updated applications and utilities to use `secrets` module instead of `random` (#6525)
* Updated locale widget to use full names for locales with shared languages (#6475)
* Removed support for runtime asset minification (#6425)
* Updated wordlists to replace potentially offensive terms (#6442)
* Fixed string localization error (#6465)
* Updated installation logic to generate Python bytecode during installation (#6591, #6602)
* Fixed new password functionality to require admins to authenticate if changing their own password (#6627)

### Operations

* Removed tracking parameters from Gitter badge in project README (#6548)
* Fixed Ansible error in Tails validation task (#6533)
* Removed usage of lsb_release in Tails and server code (#6530)
* Updated descriptions in template KeePassXC database (#6524)
* Removed old signing key and code referring to it (#6457, #6519)
* Updated contact information in kernel source offer file (#6521)

### Development

* Updated development dependencies: dparse from 0.5.1 to 0.6.2; ujson from 5.3.0 to 5.5.0 (#6565)
* Updated developer documentation URLs to point to new site at developers.securedrop.org (#6556)
* Updated dev Dockerfile to use gpgv for signature verification (#6555,#6564)
* Refactored functional tests to use fixtures throughout (#6518, #6517, #6516, #6505, #6504, #6503, #6487, #6484, #6482, #6481, #6476, #6383, #6382, #6379, #6365)
* Fixed miscellaneous typos in code comments (#6549)
* Removed grsecurity support from dev Dockerfile, misc version updates (#6528)
* Added black and isort formatting for source code (#6480, #6486)

### CI

* Added static code analysis via semgrep (#6479)
* Updated pagelayout test to address intermittent CI failure (#6470)
* Removed an external CI dependency by bundling the Codecov GPG key (#6612, #6626)

## 2.4.2

### Security

* Updated Linux kernel to version 5.15.57, which includes mitigations for the security vulnerability known as “[Retbleed](https://comsec.ethz.ch/research/microarch/retbleed/)” (#6506)

## 2.4.1

### Web Applications

* Bugfix: fixed codename filter bug affecting messages with non-ASCII characters (#6492)

### CI

* Added exclusions for 2 irrelevant safety db entries (#6473, #6477)

## 2.4.0

### Web Applications

* Refactored application CSS and updated Source Interface design (#6322, #6362, #6315, #6419, #6429)
* Updated code to remove Flask and Jinja2 deprecation warnings (#6245)
* Simplified templates using jinja2 expression-statement extension (#6378)
* Updated filesystem_id fields to be non-nullable (#6350)
* Updated gnupg commands to use direct trust model (#6397)
* Replaced potentially offensive terms in wordlists (#6402)
* Gracefully emit errors when configured languages are unavailable (#6406)
* Fixed broken link to download Tor Browser on Tor2Web interstitial (#6393, #6430)
* Add locale for Portuguese (Portugal), with the language code `pt_PT` (#6156)
* Improved 2FA token reuse protection (#6460)

### Journalist API

* Use a custom encoder to consistently format datetime objects (#6260)

### Operations

* Disabled fwupd timers to suppress inactionable OSSEC notifications (#6401)
* Updated SecureDrop release public key to version with expiry date 2020-07-04 (#6448)

### Development

* Fixed schema comparison tests (#6353)
* Refactored functional test fixtures and updated tests to use them (#6307, #6361)
* Updated safety dependency from 1.8.7 to 1.10.3 (#6396)
* Updated mypy dependency from  0.761 to 0.942 and enabled SQLAlchemy plugin (#6351)
* Removed remaining support for Xenial in package build logic (#6409)
* Added support for provisioning demo container and landing page (#6407, #6418, #6421)

### CI

* Removed logic to fetch Tor packages in nightly build (#6349)
* Replaced codecov Bash uploader with binary uploader (#6416)
* Updated CircleCI to use Python 3.8 image, GCE to use Debian 11 (bullseye) base image (#6431)

## 2.3.2

* Added Tails 5.0 compatibility updates (#6408, #6424)

## 2.3.1

* Bugfix: disabled GPG trustdb checks to prevent timeouts on operations with large keyrings (#6390)
* Updated click test dependency from 7.1.2 to 8.1.2 (#6381)
* Bugfix: fixed locale switcher on Source Interface codename page (#6377)

## 2.3.0

### Web Applications

* Added optional message filtering to allow instances to set a minimum initial message length and block initial messages containing source codenames (#6306, #6340, #6345, #6368)
* Added "skip to notification" link to allow screen-readers to navigate to flashed messages (#6336)
* Updated /generate to improve and simplify instructions on use of the codename (#6330)
* Fixed untranslated string in Read Replies widget (#6321, #6344)
* Updated Source Interface browser security level widget to reflect current Tor Browser UI (#6320)
* Added ARIA annotations for forms in Journalist Interface (#6240)
* Added tor2web detection via URL mangling (#6304)
* Removed unused bulk delete confirmation functionality (#6261)
* Added "skip to main content link" Source Interface base template (#6237)
* Added hidden antispam field to detect some automated submissions (#6302)
* Updated application to redirect to a warning page on detection of a tor2web proxy (#6300, #6335)
* Added JavaScript check to detect use of non-torified proxy servers (#6303)
* Added /robots.txt route and meta tags to disallow bots on the Source Interface (#6299)
* Fixed text overflow issue in the "Read Replies" widget (#6301)
* Updated data attributes in the codename widget to be translatable strings (#6288)
* Added support for future user-agent strings  with a 3-digit Firefox version (#6309)

### Development

* Updated `make shellcheck` to optionally use podman instead of Docker (#6239)
* Updated `make dev-tor` to set correct v3 address in Source Interface /metadata endpoint (#6308)

## 2.2.1

* Update default grsec kernel version to 5.15.26 (#6325)
* Update pillow test dependency from 9.0.0 to 9.0.1 (#6305)

## 2.2.0

### Web applications

* Refactor and simplify code for creating a new source (#6087).
* Timestamps will now be displayed using the language's "long" format (#6144).
* The "Forgot your codename?" hint will only be shown on the initial login (#6130).
* Disable caching of GPG passphrases to ensure the correct passphrase must be passed every time (#6174).
* Set minimum length requirements on HOTP and TOTP secrets (#6191).
* Improve security-related/defensive HTTP headers used in the source and journalist interfaces (#6187).
* Remove "Refresh codename" feature because of potential user confusion (#6195).
* Refactor code related to encryption (#6160).
* Stop explicitly flushing the database session (#6223).
* Upgrade to Flask 2.0 and associated refactoring (#6217).
* Improve accessibility of journalist interface by using semantic HTML5/ARIA markup (#6165).
* Information corresponding to deleted journalists will now be associated with a reserved "deleted" account internally (#6225).

### Operations

* Update grsecurity-patched Linux Kernel from 5.4.136 to 5.15.18 (#6242).
* Add support for custom hostnames to `./securedrop-admin verify` (#6153).
* Remove remnants of v2 onion services related configuration (#6169).
* Overwrite timestamps on source keys during application start (#6270).
* Dependency updates:
  * click from 6.7 to 8.0.3 (#6217)
  * flask-babel from 011.2 to 2.0.0 (#6217)
  * flask-wtf from 0.14.2 to 1.0.0 (#6217)
  * Flask from 1.0.2 to 2.0.2 (#6217)
  * itsdangerous from 0.24 to 2.0.1 (#6217)
  * jinja2 from 2.11.3 to 3.0.2 (#6217)
  * markupsafe from 1.1.1 to 2.0.1 (#6217)
  * redis from 3.3.6 to 3.5.3 (#6217)
  * rq from 1.1.0 to 1.10.0 (#6217)
  * werkzeug from 0.16.0 to 2.0.2 (#6217)
  * wtforms from 2.1.0 to 3.0.0 (#6190)
* Install dh-virtualenv from Ubuntu 21.10/Impish (#6206).
* Upgrade builder to use Rust 1.58.1 (#6234).
* Don't register trigger on Python for securedrop-app-code package (#6231).

### Development

* Improve robustness of Tor Browser version check in `make dev` (#6166).
* Remove unsupported Xenial dev Dockerfiles, templates and more. (#6166, #6185).
* Support running the dev setup using onion services with `make dev-tor` (#6167).
* Support using Podman for the development environment as an alternative to Docker (#6216).

## 2.1.0

### Web applications

* Updated HTML time tags to use valid datetime formatting (#6075)
* Refactor web applications to consolidate source user creation and session management, and remove dependencies on the system scrypt module in favour of equivalent functionality from the cryptography package (#5692, #5694, #5695)
* Updated confirmation message for successful replies in the Journalist Interface (#6102)
* Refactored Source Interface to improve accessibility, using semantic HTML and ARIA annotations (#5996, #6021, #6041, #6056, #6096)
* Increased default length of 2FA secrets from 80 to 160 bits (#5958)
* (Bugfix) Restricted length of source codenames stored in session to fit within standard session cookie (#6066)
* (Bugfix) Added a uniqueness condition for the web applications’  InstanceConfig (#5974)
* Removed the JavaScript dependency for the user deletion confirmation modal dialog on the Journalist Interface (#5696)
* Updated Source Interface to use TLSv1.3 only when HTTPS is enabled (#5988)
* (Bugfix) Removed duplicate CSS class attribute from Source Interface index page (#6049)

### Operations

* Added script to repair Tails updater for Tails versions below 4.19 (#6110)
* Silenced low-priority fwupd-related OSSEC alerts (#6107)
* Removed superfluous Tails configuration chance to enforce preservation of filenames on decompression in Nautilus (#6079)
* Removed expired signing key from Securedrop keyring (#5979, #6138)
* Added option to restore from backup file manually transferred to server (#5909)
* Added upgrade of all packages early in install, for newest ca-certificates (#6120)
* Removed version check for "verify" command for securedrop-admin (#6134)
* Dependency updates: requests 2.22.0 to 2.26.0; urllib3 1.25.10 to 1.26.6; Ansible 2.9.21 to 2.9.26 (#6046, #6109)

### Development

* (CI) Updated CircleCI configuration to use built-in branch filtering (#6086)
* Updated packaging logic to no longer treat default logo image as a conffile (#6101)
* (CI) Refactored test suite for increased parallelization (#6065, #6100, #6077)
* (CI) Added job to validate the web applications’ HTML (#6072, #6105)
* Improved the reliability of the staging provisioning playbook (#6088)
* (CI) Restricted long running staging-test-with-rebase job to run on nightlies and release candidate builds only (#6063)
* (CI) Updated updater-gui-tests job to use Python 3.7 (#6069)
* (CI) Updated CircleCI configuration to skip translation tests when not required (#6029)
* (CI) Updated CircleCI configuration to improve shellcheck filtering (#6028)
* Development dependency updates: pynacl 1.1.2 to 1.4.0; pillow 8.3.1 to 8.3.2; coverage 4.4.2 to 5.5; cryptography 3.2.1 to 3.4.7 (#6027, #6094, #6092, #5975)
* Updated package build to use latest pip (21.3) rather than system pip for virtualenvs (#6141)
* Updated QA playbook for Focal apt sources (#6123)

## 2.0.2

### Operations

* Updated default grsec kernel version from 5.4.97 to 5.4.136 (#6054)

### Development

* Updated pillow dependency from 8.2.0 to 8.3.1 (#6045)

## 2.0.1

### Operations

* Updated admin dependencies: cryptography from 3.2.1 to 3.4.7, cffi from 1.14.2 to 1.14.5 (#6030)

### Development

* Updated testinfra tests to use site-specific configuration against production instances (#6032)

## 2.0.0

### Web applications

* Removed inappropriate terms from wordlist (#5895)
* Updated dependencies: babel from 2.5.1 to 2.9.1; cryptography from 3.2.1 to 3.4.7 (#5968, #5964)
* Removed flag-for-reply functionality (#5954, #6006, #6008, #6011)
* Removed workaround for CSRF token set under Python 2 (#5957)
* Removed `source_v2_url` field from SI metadata endpoint (#5926)
* Bugfix: store directory is recreated if missing for an existing source (#5944)
* Removed Xenial-only code from codebase (#5911, #5990, #5999)
* Updated error message text (#5905, #5932)
* Updated server OS information check (#5878)
* Improved performance of logo image requests (#5874)
* Improved performance of queries for unseen submissions (#5872)

### Journalist API

* Added /sources/<source_uuid>/conversation endpoint (#5963)

### Operations

* Updated Tor version to 0.4.5.8 (#5971)
* Bugfix: Application Server submissions checker updated to run once per day (#5927)
* Removed upload-screenshots.py from application package (#5941)
* Updated dependencies: py from 1.9.0 to 1.10.0; ansible from 2.9.7 -> 2.9.21 (#5925)
* Bugfix: corrected errors in `tailsconfig` validation logic and set requirement for Tails 4.x (#5965)
* Bugfix: updater tag signature verification now matches 2021 key (#5998)
* Removed support for v2 service configuration (#5915)

### Development

* Updated mypy configuration to expect Python version 3.8 (#5977)
* Revised upgrade testing logic (#5960)
* Added Rust toolchain to the builder image (#5966)
* Removed VirtualBox support in VM scenarios (#5922)
* Improved efficiency of alembic upgrade/downgrade tests (#5935)
* Updated geckodriver and Firefox ESR versions used in tests to latest versions (#5921)
* Bugfix: corrected Ansible deprecation warning when building deb packages (#5917)
* Updated dependencies: pip from 19.3.1 to 21.1.1; pip-tools from 4.5.1 to 6.1.0; setuptools from 46.0.0 to 56.0.0; setuptools-scm from 5.0.2 to 6.0.1; pillow from 8.1.1 to 8.2.0 (#5888)
* Added codecov checksum validation, updated CircleCI machine to Focal (#5907)
* Added paxtest checks in testinfra suite, fixed tests failing under Focal (#5848, #5839)
* Bugfix: disabled errant pylint tuple definition checks (#5887)

### Translation

* Improved i18n_tool.py update-from-weblate functionality (#5863)
* Corrected untranslated messages (#5881)

### Documentation

* Bugfix: screenshots of long pages are created correctly (#5938)
* Replaced local copy of code of conduct with centralized version (#5889)

## 1.8.2

### Web applications

* BUGFIX: use post method for ossec-test-alert endpoint (#5947)

### Operations

* Add new pubkey for future release signing key rotation (#5930)
* BUGFIX: restore process no longer copies backup tarballs into memory before unpacking them (#5919)
* BUGFIX: The From: address for OSSEC alerts is now explicitly set (#5916, #5937)

## 1.8.1

### Operations

* Install a fixed version of setuptools-scm before building packages (#5877)
* Update pylint from 2.5.0 to 2.7.4, pyyaml from 5.3.1 to 5.4.1 (#5884)
* Suppress OSSEC alert caused by fwupd not being active (#5882)
* Exclude SSH onion service config from restores (#5886)
* Add support for custom logos in backups (#5880)
* Add check for SecureBoot status in installer (#5879)
* Restore playbook validates Tor config after v2 service removal (#5894)

## 1.8.0

### Web applications

* Provide end-of-life messaging and disable source interface after Xenial End-of-life (#5789)
* Adds safe deletion functionality to the Journalist Interface (#5770, #5827)
* source_app.utils.normalizer_timestamps will no longer create an empty file (#5724)

### Operations

* Support Ubuntu 20.04 Focal for SecureDrop server installs (#4728)
* Update and add annotations to SSH configuration for servers (#5666)
* (Focal) Disable v2 onion addresses on restore (#5677)
* (Focal) Replace cron-apt with unattended-upgrades (#5684)
* Remove cloud-init package during server installation (#5771)
* (Focal) Update Kernel to 5.4.97 for Focal (#5785)
* (Focal) Disable LTS upgrade prompt (#5786)
* Check for updates before most securedrop-admin commands(#5788)
* Install release-upgrader in prepare-servers role (#5792)
* (Focal) Remove aptitude and disable install-recommends(#5793)
* (Focal) Update and annotate Apache configuration (#5797)
* Update Tor to 0.4.5.6 (#5803)
* (Focal) Replace ntp and ntpdate by systemd-timesyncd (#5806)
* Use paxctl under Xenial, Paxctld for Focal (#5808)
* (Focal) Remove resolvconf (#5809)
* (Focal) Disable ipv6 via cmdline (#5810)
* (Focal) Add check to prevent Onion Service v2 install (#5819)
* (Focal) Force UTF-8 for tmux (#5844)
* (Focal): Ensure the systemd-timesyncd service is not masked (#5842)
* Update restore playbook to compare when torconfigs are being updated (#5834)
* (Focal) Ensure unattended-upgrades accepts package config defaults (#5855)

### Translations

* Update translated strings (#5790)
* Reduce localization manager review noise (#5791)

### Development

* Qualify debian packages with target distribution (#5765)
* Always update Tor Browser to the latest release (#5774)

## 1.7.1

* Bugfix:  Fixes an issue affecting availability of Source and Journalist Interfaces for a subset of instances. (#5759)

## 1.7.0

### Web Applications

* Add Simplified Chinese to our supported languages (#5069)
* Enhance UI for account creation (#5574)
* Improve reminder on Journalist Interface to enable v3 onion services (#5679)
* Prevent unhandled exception and add error message in Journalist Interface when download fails due to missing file (#5549, #5573, #5733)
* No longer show session expiration on source interface when not logged in  (#5582)
* Allow admins to set org name on Journalist Interface (#5629)
* Ensure Tor Browser warning appears on Source Interface for MacOS users (#5647)
* Prevent unhandled exception when submitting a password quickly multiple times (#5618)
* No longer call the SecureDrop Submission key a journalist key on Source Interface (#5651)
* Update navigational symbol to migrate to previous page on Journalist Interface (#5641)

### Operations

* Prepare for upcoming support for Ubuntu 20.04 (Focal) (#5527, #5529, #5538, #5544, #5556, #5559, #5581, #5602, #5604, #5608, #5614, #5615 #5619, #5669, #5674, #5691)
* Update Tor to 0.4.4.6 (#5648)
* Add daily OSSEC alert if v2 onion services are enabled (#5673, #5682)
* OSSEC alert improvements and fixes (#5665, #5330)
* Fix failure during `securedrop-admin install` when Ansible config is stale (#4366)
* Update dependencies (#5585, #5612, #5623, #5671)

### Translations

* Allow more strings to be translated (#5567, #5707)
* Ensure translators get credit for translations (#5571)

### Developer-Only

* Migrate documentation to new repository (#5520, #5583, #5587)
* Add more type annotations (#5531, #5532, #5533, #5534, #5536, #5547, #5597, #5589, #5600, #5616)
* Continuous integration updates (#5542, #5550, #5658)
* Add file attachments and seen records when generating test data (#5580, #5645)
* Improve functional tests (#5590, #5610, #5621, #5631, #5634)
* Improve test coverage (#5632, #5644, #5657, #5662)
* Refactor and improve testing during development (#5558, #5575, #5686, #5683, #5595)
* Update Xenial builder image (#5561) and vagrant boxes for SecureDrop 1.6.0 (#5569)
* Fix SSH connection failure in Molecule upgrade script (#5654)
* Replace FontAwesome glyphs with PNGs (#5593, #5625, #5628)

## 1.6.0

### Web Applications

* Adds error handling for file deletion operations (#5549)
* Update source Interface to render /lookup if replies are missing (#5497)
* Remove traces of obsolete dependencies from virtualenv (#5487)
* Internationalize "Show Password" on Journalist Interface (#5483)
* Remove duplicated route (#5440)

#### Journalist API

* Add /seen endpoint (#5513)
* Add /users endpoint (#5506)
* Add seen tables and update seen status based on downloads in JI (#5505)

### Operations

* Added 30 sec delay for post-reboot checks (#5504)
* Update Tor to 0.4.4.5 (#5502)
* Use built-in .venv module in dh_virtualenv to build application code package (#5484)
* Update cffi and argon_cffi (#5458)

### Documentation

* Add note on creating virtualenv and Fedora install docs (#5478)
* Fix capitalization for readability (#5447, #5448)
* Add instructions to update public docs during release (#5441)

### Development

* Add preliminary support for testing Ubuntu Focal (#5518, #5515, #5486, #5482, #5465, #5444)
* Add type annotations to journalist app (#5464, #5460)
* Add no_implicit_optional in mypy config (#5457)
* Adds QA playbook test a production instance (#5452)
* Add Weblate screenshot uploader tool (#5450)
* Updated testinfra tests to optionally run against a production instance (#5318)

## 1.5.0

### Web Applications

* Fixes #5378, adds more checks for invalid username (#5380)
* Add v2 onion service deprecation warning to journalist interface (#5366, #5726)
* Changed modal_warning message in the delete source dialogue to match the context (#5358)
* Further redaction of wordlist (#5357)
* Handle case of deleted journalists (#5284)

### Operations

* Remove unused/disabled source onion service info files (#5404)
* Update Tor to 0.4.3.6 (#5374)
* Add v2 onion service deprecation warning to securedrop-admin (#5370)
* Update kernel version to 4.14.188 (#5365)
* Ignore tmp files in /var/lib/securedrop/shredder (#5308)

### Journalist API

* Handle disconnected replies in API endpoint (#5351)
* Handle disconnected submissions in API endpoint (#5345)

### Documentation

* Consistent usage of "onion services" terminology (#5379)
* Add v2 deprecation timeline & warnings to docs (#5373)

### Development

* Add type annotations for source_app/utils.py (#5300)
* Update syntax for existing type annotations (#5298)

## 1.4.1

* Bugfix: Updated securedrop-admin utility to validate instance configuration correctly when v3 onion services are disabled (#5334)

## 1.4.0

### Web Applications
* Moved source deletion to a background process to resolve timeout issue (#2527)

### Operations
* Updated Tor from 0.4.2.7 to 0.4.3.5 (#5292)
* Added check for nameserver configuration loaded during install (#5288)
* Added system configuration check script to securedrop-ossec packages (#5287)
* Updated SecureDrop Release signing key so that it doesn't expire until June 30, 2021 (#5277)

### Documentation
* Added instructions for password rotation to Admin Guide (#5301)
* Updated references to uncommon OSSEC alerts (#5316)


## 1.3.0

### Web Applications

* Use WTForm for source interface submission form (#5226)
* Updated behavior of Logout button, adding a dedicated logout page with directions for sources on wiping Tor Browser session data (#5116)
* Changed references to “journalists” in Source Interface text to “teams”  (#5175)
* Updated Source Interface to create a single source codename when multiple  “/generate” tabs are opened in a single session (#5075)
* Bugfix: Added Journalist Interface error templates to Apache2 AppArmor profile (#5086)
* Updated 2FA instructions to use consistent nomenclature across the Journalist Interface (#5049)
* Added v2 and v3 Source Interface onion addresses to Source Interface `/metadata` endpoint (#5074)
* Added confirmation message when document submission preference changed in Journalist Interface (#5046)
* Updated Source Interface `/lookup` design (#5096)


### Journalist API

* Improved response time of `/get_all_sources` endpoint by caching source public keys (#5184)
* Bugfix: Updated `/replies` endpoint to correctly return replies associated with a deleted journalist account (#5178)

### Operations

* Updated Ansible version from 2.7.13 to 2.9.7 (#5199)
* Updated OSSEC version from 3.0.0 to 3.6.0 (#5196)
* Updated Tor version to from 0.4.1.6 to 0.4.2.7 (#5192)
* Updated default Grsecurity-patched kernel version from 4.14.154 to 4.14.175 (#5188)
* Updated Ansible timeout value to 120 seconds and simplified server reboot handling (#5083)

### Admin Tails workstation

* Bugfix: Updated GUI updater to return correct error when Tails configuration script times out (#5169)
* Updated GUI updater to log and exit duplicate instances (#5067)
* Added option to preserve Tor service configuration when restoring from backups (#5115)


### Developer Workflow

* Improved QA loader script to produce datasets more consistent with production data (#5174, #5200)
* Updated SecureDrop Core’s Qubes staging environment provisioning, removing requirement for manual reboots and renaming VMs to avoid conflicts with SecureDrop Workstation (#5190, #5099)
* Dependency updates:
  *  pyyaml from 5.1.2 to 5.3.1;
  * urllib from 1.25.3 to  1.25.8;
  * safety from 1.8.4 to 1.8.7;
  * pillow from 6.2.1 to 7.0.0;
  * mypy from 0.701 to 0.761;
  * pylint from 1.8.1 to 2.4.4;
  * markupsafe from 1.0 to 1.1.1;
  * setuptools from 41.6.0 to 46.0.0
  * astroid from 2.3.3 to 2.4.0
  (#5182, #5151, #5133, #5219)
* Updated makefile lint target to ignore SC2230 shellcheck warnings (#5171)
* CI: Updated translation tests to run in parallel across set of supported languages (#5062)
* Added HTML markup to test submissions in dev environment (#5068)
* Disabled `build-debs` builder update test for non-RC branches (#4988)
* CI: Added LGTM configuration file (#5076)
* Added support for running multiple simultaneous dev containers (#4633)
* CI: Increased Docker verbosity, fixed image caching (#5113)
* Added `testinfra` makefile target, to run infra tests against a staging instance (#5114)
* Added functionality to load test datasets during staging provisioning (#5143)

### Documentation

* Updated Ubuntu verification instructions in install documentation (#5098, #5106)
* Added journalist/admin off-boarding guide (#5012)
* Added server BIOS update guide (#4991)
* Miscellaneous documentation updates (#5059, #5063, #5066, #5072, #5078, #5138)

## 1.2.2

### Web Applications

* Update psutil to 5.7.0

### Admin Tails workstation

* Pin setuptools to requirements file (#5159)

## 1.2.1

### Web Applications

* Updated Tor Browser user agent detection (#5087)
* Added caching of source keys (#5100)
* Removed the ability to change source codenames/designations (#5119)

## 1.2.0

### Web Applications

* Added option to disable document uploads on Source Interface (#4879).
* Updated alt text for Source Interface logo image (#4980).
* Added journalist name fields to API '/token' response (#4971).

### Operations

* Updated submission cleanup queues to be managed with systemd instead of supervisor (#4855, #5037).
* Updated grsecurity kernels to version 4.14.154 to mitigate CVE-2019-11135 (#4962, #4990).

### Admin Tails workstation

* Cleaned up unused code in securedrop-admin tool (#4933).
* Updated network hook to use Python 3 (#5039)

### Developer Workflow

* Added SECURITY.md to the repository (#4994).
* Fixed Journalist Interface functional tests that require specific Tor Browser security settings (#4987, #4995).
* Added developer quickstart instructions to the repository README (#4983, #4984).
* Fixed admin pip requirements updater (#4955).
* Added an environmental variable to control Docker build verbosity (#4943, #4974).
* Increased libvirt prod VM memory to 1024MB (#4918).

### Documentation

* Updated automatic screenshots to use Tor Browser for Source Interface (#4975).
* Added auto-cropping for automatic screenshots (#4958).
* Updated Source Guide (#4880).
* Updated documentation to refer to Tails 4.0 (#4937, #4942, #4998, #5035)
* Miscellaneous documentation fixes (#4922, #4957, #4960, #4968, #4970, #4986, #5002).

## 1.1.0

### Web Applications

* Added Czech to supported languages (#4885).
* Added Slovak to supported languages (#4940).
* Reordered and updated wording of manage.py (#4850, #4858).
* Remove python 2 support in SecureDrop server code (#4859).

### Operations

* Update Tor to 0.4.1.6 (#4848).
* Updated dependencies for app server and admin workstation (#4865, #4884).
* Fixes to packaging (#4870, #4871).
* Improve specification of securedrop-app-code dependencies (#4876).

### Admin Tails workstation

* Added StartupNotify=true to desktop shortcuts (#4841).
* Adds workarounds for Tails 4.0 detection (#4852).
* Moves securedrop-admin into Python 3 (#4867).
* Enables Source and Journalist desktop icons in Tails 4 (#4872).
* Added check for admin password in GUI updater (#4877).

### Documentation

* Transfer device and export recommendations (#4838).
* Miscellaneous documentation fixes (#4844, #4853, #4874, #4886).

### Web Applications

* Reordered and updated wording of manage.py (#4850, #4858).
* Remove python 2 support in SecureDrop server code (#4859).

### Operations

* Updated dependencies for app server and admin workstation (#4865, #4884).
* Fixes to packaging (#4870, #4871).
* Improve specification of securedrop-app-code dependencies (#4876).

### Admin Tails workstation

* Added StartupNotify=true to desktop shortcuts (#4841).
* Adds workarounds for Tails 4.0 detection (#4852).
* Moves securedrop-admin into Python 3 (#4867).
* Enables Source and Journalist desktop icons in Tails 4 (#4872).
* Added check for admin password in GUI updater (#4877).

### Documentation

* Transfer device and export recommendations (#4838).
* Miscellaneous documentation fixes (#4844, #4853, #4874).

## 1.0.0

### Web Applications

* UI: refresh source and journalist interface design (#4634, #4666).
* Update copyright dates (#4638).
* Update language selector design in the menu to be more accessible (#4662).
* Add commands to manage.py for admins to detect and correct deletion issues (#4713).
* Use shred instead of srm to securely delete files (#4713).
* Bug fix: Invalidate Session When Admin Resets Journalist Password (#2300).
* Bug fix: Interrupted deletion jobs are now resumed on reboot (#4713).
* Bug fix: Clean up any orphaned submissions/replies where source has already been deleted (#4672).
* Bug fix: Resolve a bug with the “Select unread” feature on the journalist interface (#4654).

### Operations

* Use dh-virtualenv and mod_wsgi to create securedrop-app-code package, run Python 3 version of the web applications on instances (#4622).
* Adds support for v3 onion services for SecureDrop source, journalist, and SSH interfaces (#4652, #4710, #4690, #4675).
* Adds warning in securedrop-admin sdconfig if v3 onion services and HTTPS on the source interface are both enabled (#4720).
* Uses latest Tor series (0.4.x) instead of LTS (0.3.x) series (#4658).
* Move tasks removing old kernels to common Ansible role (#4641).

### Documentation

* Update translator documentation (#4719).
* Fix incorrect alias name in firewall documentation, update screenshots (#4685).
* Remove redundant Tails guide (#4673).
* Remove old printer troubleshooting guide (#4651).

## 0.14.0

### Web Applications

* Add support for setting journalist names, and expose via journalist API (#4425, #4459).
* Update instructions for sources regarding Tor Browser 8.5 security settings (#4462,  #4494).
* Replace cloud icon with download icon in source interface (#4548).
* Expose supported locales in source interface metadata API (#4467).
* Remove unnecessary FontAwesome CSS from source interface (#4464).
* Bug fix: If sessions expire on /generate on the source interface, redirect to index (#4496).
* Add explanatory text for authenticator reset buttons in Journalist interface (#3274)

### Operations

* Use archive module for securedrop-admin logs command (#4497).
* Fix Ansible deprecation warnings (#4499).
* Update grsecurity kernels to version 4.4.182 (#4543).
* Add intel-microcode as dependency (#4543).
* Update securedrop-keyring to 0.1.3: update expiration of signing key and add uid (#4578)
* Switch to keys.openpgp.org as the default keyserver (#4576)
* Update securedrop-admin tool to use only hkps://keys.openpgp.org when retrieving release key (#4585)

### Developer Workflow

* Add new functional test of /metadata endpoint (#4536).
* Add workaround for Circle CI’s problems with branch filtering  (#4505).
* Automatically rerun flaky admin tests (#4466).
* Improve localization manager documentation and update script used for gathering translator names (#4493, #4482, #4469).

### Documentation

* Update SecureDrop screenshots for source, journalist and admin guides (#4564)
* Update Admin Workstation setup instructions to use keys.openpgp.org (#4586)

## 0.13.1

* Fix download of Journalist GPG key via Source Interface (#4523)

## 0.13.0

### Web Application

* Added fingerprint of GPG reply key to Source object in Journalist API (#4436)
* Updated message for flagged sources in Source Interface (#4428)
* Added type hinting in Journalist Interface (#4404, #4407)
* Added a /logout endpoint to the Journalist API (#4349)
* Fixed 500 error caused by inconsistent session (#4391)
* Added submission file sha256 hash as Etag on downloads in Journalist Interface (#4314)
* Added Python3 compliance for Source and Journalist Interface (#4239)
* Updated Python cryptography dependency, updated safety check (#4297)
* Added support for HTTP DELETE method on Journalist Interface for use by API (#4023)
* Removed cssmin dependency (#4227)
* Removed jQuery dependency from Journalist and Source Interface (#4211)

### Operations
* Removed support for Ubuntu 14.04 LTS (Trusty) (#4422, #4416, #4348, #4311, #4224)

### Developer Workflow

* Updated functional tests to run against Tor Browser (#4347)
* Consolidated CI lint Makefile targets (#4435)
* Added 0.12.2  boxes for use with the Molecule upgrade scenario (#4393)
* Added deb tests to builder image update (#4388)
* Removed unused Jenkins configuration (#4337)
* Added version pinning for Tor package fetch CI job (#4300)
* Updated Tor version check to 0.3.5.8 (#4258)
* Updated staging CI job timeout to 20 minutes (#4218)
* Added CI job to run page layout tests in all supported languages for i18n-\* branches (#4184)
* Fixed `bandit` Makefile target (#4429)
* Added multiple locale settings in staging environment (#4419)
* Added hypothesis support and example hypothesis test on encrypt/decrypt functionality (#4412)
* Added support for asynchronous jobs in dev container (#4392)
* Updated Qubes staging environment to use Xenial by default (#4344, #4228)
* Updated dev environment to use Xenial by default (#4213)
* Fixed Dockerfile apt caching error, fixed error in create_dev_data.py (#4353)
* Added support for use of VNC during functional tests (#4288, #4324)
* Added support for staging-specific data to create-dev-data.py (#4298)
* Removed firefox and other packages from app-test Ansible role (#4277)
* Added option to control number of sources created in a dev environment (#4274)
* Added check to ensure dev virtualenv uses Python 2 (#4127)

### Documentation

* Added instructions on backing up and restoring workstations with rsync (#4402)
* Updated references to 4-port firewall in server setup docs (#4430)
* Standardized references to Submission Key (#4413)
* Updated links to Tails documentation (#4409)
* Clarified use of KeePassX in passphrases documentation (#4368)
* Renamed "Terminology" section to "Glossary" (#4405)
* Added instructions for upgrading to Xenial after Trusty EOL (#4395)
* Clarified workstation hardware requirements (#4369)
* Updated logging documentation (#4359)
* Updated release management documentation (#4386)
* Updated nginx landing page example and journalist onboarding guide (#4370)
* Added attacks and countermeasures to SecureDrop threat model document (#4244)
* Added recommendation to backup workstations (#4302)
* Updated recommended Apache configuration in landing page guide (#4238)
* Updated pronouns in journalist guide (#4254)
* Updated README to make it easier to credit i18n contributors (#4243)

## 0.12.2

### Web Application

* Remove NoScript upload instructions on Source Interface (#4160)
* Disable Source Interface on instances running Trusty after April 30th (#4325)
* SecureDrop application dependencies have been updated (#4346)

### Operations

* SecureDrop grsec kernels have been updated to 4.4.177 and provide support for Intel e1000e series NICs (#4308)
* OSSEC test notifications will now generate ossec alerts (#4340)
* Updated AppArmor for Apache2 (#4362)

### Tails Environment

* Backup script should now more reliably download large backups from the app server (#4326)
* SecureDrop GUI updater should now be limited to a single running instance (#4309)

### Documentation

* Instruct admin to look up latest Tails version for Xenial upgrade (#4325)

## 0.12.1

### Web Application

* Add "Back to submission page" link to NoScript docs (#4208)

### Operations

* Ensured WiFi related packages are not installed on Xenial on upgrade (#4163)
* Try harder to attach to a `tmux` session on upgrade (#4221)
* Control locale during Ansible runs (#4252)

### Tails Environment

* Resolved error in GUI updater due to flaky keyservers (#4100)

### Documentation

* Add documentation indicating fresh installs should be done on the Ubuntu 16.04.6 iso (#4234)

## 0.12.0

### Web Application

* Add Romanian and Icelandic as supported languages (#4187)
* Added toggle to show password for journalists on login (#3713)
* Updated language referencing Tor button (#4131, #4141)
* Added instructions for disabling NoScript XSS because of upload problem (#4078, #4159)
* Prevented setting session cookies on API endpoints (#3876)
* Updated API to allow clients to set a reply's UUID (#3957)
* Changed GPG key generation to avoid leaking key creation date (#3912)
* Fixed race condition that caused all public keys to be exported by API (#4005)
* Added `filename` to payload when creating a reply via the API (#4047)
* Fix bug that caused internal server errors on malformed auth tokens (#4053)
* Added alert on journalist interface to alert when the operating system is out of date (#4027)
* Added journalist UUID to payload when creating an auth token via the API (#4081)
* Added GPG 2.1+ compatibility (#3622, #4013, #4038, #4035)
* Added OS information to metadata endpoint (#4059)

### Operations

* Removed hardcoded Ansible plugin `profile_tasks` (#2943)
* Fixed restore logic to ensure recreation of onion services (#3960, #4136)
* Added logic to conditionally update the `release-upgrades` prompt (#4104, #4142)
* Added logic to ensure packages required by Ansible are present on Xenial systems (#4109, #4143)
* Ensured Tor is installed from FPF repo (#4175, #4169)
* Set Debian packages to only use explicitly declared conffiles (#4176, #4161)
* Removed `iptables` UID restrictions to allow `apt` to work correctly (#3952)
* Updated kernels to 4.4.167 and removed wireless support (#2726)
* Updated `cron-apt` remove action to occur after security (#4003)
* Updated AppArmor profile for Xenial (#3962)
* Removed common-auth PAM customizations (#3963)
* Updated `./securedrop-admin logs` command to log installed packages (#3967) and `cron-apt` logs (#4000)
* Explicitly declared onion services as v2 (#4092)
* Added ability to store both Trusty and Xenial Debian packages (#3961)
* Added ability to fetch upstream Tor Debian packages for inclusion in FPF repo (#4101)
* Run `haveged` confined on Xenial (#4098)
* Updated PaX flag management for on Apache on Xenial (#4110)

### Developer Workflow

* Fixed the QA data loader to prevent clobbering data (#3793)
* Fixed updated version script (#4146)
* Added nested virtualized to CI (#3702)
* Moved to Vagrant 2.1.x (#3350)
* Fixed linting tasks on macOS (#3996)
* Added automatic re-running of flaky admin tests (#4004)
* Increased max line to 100 characters for Python files (#4006)
* Re-added static analysis and Python dependency checking to CI (#4033)
* Updated setuptools in CI (#4036)
* Added Trusty and Xenial test targets to CI (#3966)
* Moved CI nightly jobs to 4AM UTC (#4067)
* Fixed bug where failed CI runs were reported as successes (#4066)
* Fixed Xenial-specific errors in tests (#4037, #4039)
* Added 0.11.1 upgrade testing boxes (#4093)
* Ensured config test coverage on Xenial (#3964)

### Documentation

* Added documentation on how to test upgrades (#3832)
* Added documentation on how to set the locales (#3846)
* Added documentation for upgrading from 0.10.0 to 0.11.x (#3982)
* Added documentation on how to prepare the app and mon servers for upgrade to Xenial (#4044)
* Updated date where SecureDrop uses Ubuntu Trusty as default OS (#4062)
* Updated list of hardware recommendations to remove Gigabyte BRIX (#3197, #4075) and added updates to NUCs and Mac Minis (#3976)
* Added note about how dev can generate 2FA tokens (#4095)
* Removed old markdown redirect (#4097)
* Updated SecureDrop client references (#4102)

## 0.11.1

### Operations

* Security bugfix: Upgrade apt without following redirects on first install, addresses CVE-2019-3462 (#4061)

## 0.11.0

### Web application

* UX improvement: Add username constraints to admin user addition page text (#3746)
* UX improvement: Removed screensaver from the source interface (#3722)
* Update icons on source interface (#2508, #3809)
* Replace 'administrator' with 'admin' in the journalist interface (#3940)
* API: Posting replies returns uuid (#3915)
* Bugfix: Save only the base filename in DB for replies using API (#3918)

### Operations

* Bugfix: resolve OSSEC GPG key import issue in Ansible (#3928)
* Updated Ansible to 2.6.8 (#3945)
* Update grsecurity kernels to 4.4.162 (#3913)
* Security bugfix: Disable unnecessary sshd config options (#3979)
* Removes 3.14.x grsecurity kernels (#3913)

### Developer Workflow

* Updated instruction to merge translations to & from the new Weblate (#3929)
* Updated requests module to 2.20.0 (#3908)
* Updated copyright statement to acknowledge all contributors (#3930)
* Added initial client documentation (#3922)

## 0.10.0

### Web Applications

* UX: modify size of input boxes to accommodate long passphrases (#3761)
* Security bugfix: sources can no longer delete replies for other sources if they know the target filename (#3892)

### Operations

* Updated OSSEC to 3.0.0 and use GPG signatures for verifying sources (#3701)
* Update paramiko to 2.4.2 (#3861)
* Enforce use of the latest grsecurity-patched kernel on servers (#3842)

### Development

* Xenial transition: Enable clean install in Xenial staging environment (#3833)
* Document Molecule upgrade scenario (#3720)

## 0.9.1

* Bugfix: Resolve error in SecureDrop Updater due to incorrect working directory (#3796)

The issues for this release were tracked in the 0.9.1 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestone/47

## 0.9.0

### Web Applications

* Argon2 is now used to hash journalist passwords (#3506)
* Preserve conversation history only for journalists (#3688, #3690)
* Journalist interface can now be accessed via API (#3700)

### Operations

* Updated the grsecurity-hardened Linux Kernels to 4.4.144 for app and mon servers (#3662)
* Updated Tor to version 0.3.3.9 (#3624)
* Updated Flask to 1.0.2 and Werkzeug to 0.14.1 (#3741)
* Updated securedrop-keyring package to 0.1.2 (#3752)

### Tails Environment

* Updated cryptography to 2.3 (#3679)

### Developer Workflow

* Dev container: Docker ports are now mapped only to localhost (#3693)

### Documentation

* Miscellaneous documentation improvements (#3623, #3624, #3655, #3670, #3717, #3710)

The issues for this release were tracked in the 0.9 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestone/44


## 0.8.0

### Web Applications

* Adds a new supported language: Swedish (#3570)
* Replace PyCryptodome with pyca/cryptography (#3458)
* Add explanatory text to source interface screensaver (#3439, #3455)
* Rename "Delete collection" on journalist interface for clarity (#2419)
* Security: Login throttling is now per-user, mitigating a DoS vector (#3563)

### Operations

* Updated the grsecurity-hardened Linux Kernels to 4.4.135 on app and mon servers (#3494)
* Fixed race condition with journalist notification where a daily notification would not be sent (#3479)
* Removed 2FA for console logins for app and mon servers (#3507)
* Suppresses OSSEC alerts asking SecureDrop administrators to upgrade to Xenial (#3205)
* Upgraded to Tor 0.3.3.7 on app and mon servers (#3518)
* Bugfix: Enable mon and app servers to use /32 addresses (#3465)

### Tails Environment

* Support conditional questions in securedrop-admin sdconfig (#3401)
* Security: Ensure that a branch cannot accidentally be checked out after tag verification (#3567)

### Developer Workflow

* Use Alembic to version control database schema (#3211, #3260, #3273)
* Integration tests for securedrop-admin sdconfig (#3472)
* Use pytest for more Journalist Interface unit tests (#3456)
* Rename Document Interface to Journalist Interface in Docker development environment (#3397)

### Documentation

* Miscellaneous documentation improvements (#3404, #3405, #3431, #3435,#3437, #3440, #3457, #3463, #3467, #3468, #3476, #3480)

The issues for this release were tracked in the 0.8 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestone/43

## 0.7.0

### Web Applications
* Updated messages on source interface (#3036, #3132, #3321, #3322)
* Use io.open instead of open across the web applications (#3064)
* Include token reuse under login hardening flag (#3175)
* Add Orbot warning in source interface (#3215)
* Removed compression time metadata on submission gzip archives (#3305)
* Resolve HTTPS CSRF validation failure due to Referrer-Policy on source interface (#3351)

### Operations
* Admins can optionally enable a daily encrypted email sent to journalists indicating whether or not they should check SecureDrop (#1195, #2803)
* Admins can optionally disable the use of Tor for SSH and enable SSH over local network (#2592)
* Remove vanilla Xenial Ubuntu kernels at install-time (#3158)
* Added daily_reboot_time option in sdconfig (#3172)
* Removes duplicate tor-apt repo from apt config (#3189)
* Remove installed vanilla kernels via cron-apt (#3196)
* Fixed typo in ansible restore script (#3263)

### Tails Environment

* Securedrop-admin update: Try alternate keyserver if primary is not available (#3257)
* Fixed typo in ansible restore script (#3263)
* Add SecureDrop Administrator Workstation updater GUI (#3300)
* Add HTTPS-related variables to securedrop-admin sdconfig prompt (#3366)
* Improved tag validation logic for securedrop-admin update (#3406)

### Developer Workflow

* Add test data to Docker development environment (#3081)
* i18n: centralize the list of supported languages in ./i18n_tool.py (#3100)
* The Vagrant development environment VM has been deprecated (#3271). The current supported development environments are as follows:
** The Docker development environments for application code changes and securedrop-admin changes.
** Staging or production Vagrant VMs for all other changes.
* Improved application tests by using pytest fixtures (#3140, #3179, #3181, #3183, #3190, #3191, #3249, #3250, #3256)
* Unit tests: Verify journalist username containing whitespace can login (#3218)
* Added hashes for pluggy python wheels (#3272)

### Documentation
* Update branch management docs (#3171)
* Add Release manager guide (#3202)
* Miscellaneous documentation improvements (#3099, #3147, #3153, #3156, #3168, #3201, #3252, #3265, #3295, #3315, #3359).

The issues for this release were tracked in the 0.7 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.7.

## 0.6

### Web Applications

* Adds CSS-based source deletion confirmation (#295 ,#2946).
* Resolve responsive issues on source and journalist interfaces (#2891, #2974).
* Remove config global state (#2969).
* Add functional test coverage for all JavaScript functionality (#2405).
* Migrate to Flask-SQLAlchemy (#2866).
* Improve UX on admin logo uploads (#2876).
* Enable vacuum and secure delete in the database (#2868).
* Bugfix: Ensure session is available for async_genkey (#2988).
* Bugfix: Fix user confirmation before 2FA reset (#2920).

### Operations

* Update grsecurity-hardened kernels to 4.4.115 (#3077).
* Bump Ansible version to 2.4.2 (#2929).
* Allow sasl_domain to be empty (#2482).
* Bugfix: Update AppArmor rule for Apache (#3020).
* Allow syscheck to monitor /var/lib/tor/services (#2960).

### Tails Environment

* Add commands to check for and apply updates to the securedrop-admin CLI (#2976).

### Developer Workflow

* Make the Docker-based development environment the default (#2902).
* Rebase branches prior to running CI jobs (#2934).
* Implement dev-shell script for interactive execution of commands in development containers (#2861, #2956).
* Adds Bandit static analyzer for automated security scanning (#3055).
* Adds initial support for checking type annotation (#3001, #3006, #3032).
* Adds test user to development environment (#3040).
* Update release script for new release candidate policy (#2950).

### Documentation

* Miscellaneous documentation improvements.

The issues for this release were tracked in the 0.6 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.6.

## 0.5.2

* Replace PyCrypto (#2903).
* Use `max_fail_percentage` to force immediate Ansible exits in playbook runs (#2922).
* Bugfix: Dynamically allocate firewall during OSSEC registration (#2748).
* Bugfix: Add all languages to sdconfig prompt (#2935).

The issues for this release were tracked in the 0.5.2 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestone/41

## 0.5.1

### Web Applications

* Add Arabic, Chinese, Turkish and Italian translations (#2830).
* Enable administrators to update the SecureDrop logo via the Admin Interface (#2769).
* Enable administrators to send test OSSEC alerts via the Admin Interface (#2771).
* Improve the language menu (#2733).
* Disable .map file generation when compiling CSS files from SASS (#2814).
* User logout on password reset (#2756).
* Update pip requirements (#2811).

### Operations

* Use apt mirror for Tor packages (#2441).
* Push grsecurity tasks to the beginning of the install process (#2741).
* Remove flaky SMTP/SASL host validation during SecureDrop configuration (#2815).
* Automate post-Ubuntu install checks verifying DNS, NTP is working (#2129).
* Validate GnuPG key is not a private key prior to install (#2735).
* Remove HMAC-SHA1 from SSH config (#2730).

### Monitoring

* Prevent OSSEC from sending daily emails on startup (#2701).

### Documentation

* Replace Google Authenticator with FreeOTP (#2757).
* Add recommended landing page content (#2752).

The issues for this release were tracked in the 0.5.1 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.5.1.

## 0.5

### Web Applications

* Internationalize both web applications (#2470, #2392, #2400, #2374, #2626,
  #2354, #2338, #2333, #2229, #2223).
* Localize in Dutch, French, German, Norwegian, Portuguese and Spanish.
* Add language picker to web applications (#2557).
* Refactor both web applications using Flask Blueprints (#2294).
* Add default 120 minute session timeout on both interfaces (#880, #2503).
* Only show source codename on first session (#2327).
* Invert `login_required` decorator on journalist interface so that logins
  are required by default (#2460).
* Require entry of old password before changing password (#2304).
* Use whitespace control on Jinja templates (#2413).
* Add reset icon to reset password button (#2423).
* Improve form validation on new user creation in journalist interface (#2500).
* Improve form validation on login form on source interface (#2376).
* Resolve confusing use of first/third person on user creation (#2323).
* Show which journalist is logged in on the journalist interface (#2293).
* Create friendly session expiry page (#2290).
* Improve UX to get to individual source page on journalist interface (#2130).
* Improve UX on login forms by making fields longer (#2288).
* Bugfix: Fix input validation on Yubikey for 2FA HOTP (#2311).
* Bugfix: Remove extra level in folders in submission downloads (#2262).

### Operations

* Allow apache/apparmor file exception for proving onion ownership to a CA (#2602)
* Enable admins to set supported locales via SecureDrop admin script (#2516)
* Update AppArmor rules for Apache (#2507).
* Reduce number of pip requirements files (#2175).

### Monitoring

* Add `/boot` to integrity checking (#2496).
* Bugfix: Remove OSSEC syscheck monitoring of temporary files produced by bulk
  download (#2606).

### Tails Environment

* Bugfix: Use host for SASL and SMTP domain validation (#2591).
* Bugfix: Add trusted metadata to SecureDrop .desktop files (#2586).

### Developer Workflow

* Add updated Data Flow Diagrams (#2370).
* Add safety check for Python dependencies in CI (#2451).
* Remove noisy GnuPG debug output on test failure (#2595).
* Convert tests in SubmissionNotInMemoryTest to pytest (#2548).
* Document virtualized Admin Workstation setup (#2204, #2607).
* Bugfix: Remove extra `rqworker` start in unit tests (#2613).
* Bugfix: Resolve test failures in VirtualBox (#2396).

### Documentation

* Add sample SecureDrop privacy policy to documentation (#2340).
* Break out "Deployment Best Practices" into discrete docs section (#2339).

The issues for this release were tracked in the 0.5 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.5.

## 0.4.4

Bugfix release. Fixes configuration management logic to ensure all packages
are properly validated prior to installation.

* Remove force=yes in package install tasks in Ansible config.
* Upgrade Ansible to 2.3.2 to address CVE-2017-7481.
* Add securedrop-admin `logs` command for collecting log files.
* Increases expiration date on SecureDrop Release Signing Key to October 2018.

Since this is a bugfix release, the changes on the 0.4.4 milestone
are not included here. Those issues have been postponed to a future release.

## 0.4.3

The issues for this release were tracked in the 0.4.3 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.4.3.

### Web Applications

* Automatically generate diceware passphrases for SecureDrop journalists and administrators (#980).
* Add minimum length check for journalist usernames (#1682).
* Resolve inconsistently named source IDs (#1998).
* Fix transient test errors (#2122, #2214, #2205).
* Progress towards internationalizing SecureDrop: Add utilities for extracting strings for translation and for compiling translations (#2120), add DEFAULT_LOCATE to config.py (#2197).
* Make source interface index responsive (#2235).
* Remove safe filter from jinja templates (#2227).

### Operations

* Retry SMTP relay and SASL domain verification tasks in validate role (#2099).
* Disable Postfix in the staging environment (#1164).
* Refactor Debian package build logic (#2160).
* Reboot staging servers on first provision (#1704).
* Bugfix: Whitelist fontawesome-webfont.svg in AppArmor (#2075).

### Tails Environment

* Bugfix: Use .xsessionrc in SVS to prevent unpacking of office files (#2188).

### Monitoring

* Remove OSSEC alert for Tor Guard overloaded log event (#1670).
* Remove OSSEC alert when journalists bulk delete submissions (#1691).
* Remove OSSEC alert for OSSEC keepalive event (#2138).
* Add "Ansible playbook run on server" OSSEC rule (#2224).
* Update locations of logs on app server for OSSEC syscheck (#2153).
* Adds regression testing for OSSEC false positives (#2137).

### Developer Workflow

* Adds flake8 linting (#886).
* Add code style guide for contributors (#47).
* Produce screenshots before and after Selenium tests for debugging (#2086).
* Add HTML linting (#2081).
* Add Makefile targets for linters (#1920).
* Adds "page layout" automated tests that take screenshots of the source and journalist interface in each language (#2141).
* Expose documentation on port 8000 on development machine (#2170).
* Make Makefile self-documenting (#2169).
* Add pre-commit hook for developers (#2234).
* CI: Do not run staging CI if only documentation has changed (#2132).
* CI: Use new Trusty image for Travis CI (#1876).

### Documentation

* Adds guide and glossary for translators (#2039, #2162).
* Adds i18n guide for developers (#2118).
* Adds note in Tails upgrade documentation about missing "Root Terminal" launcher in workstation backup procedure (#2065).
* Update AppArmor developer documentation (#2077).
* Adds pre-installation checklist (#2139).
* Specify NTP server to use in install documentation (#2094).
* Adds new tested printer for use in SecureDrop airgap (#2117).
* Add documentation for multiple administrators managing config (#2096).
* Update KeePassX (#2158).
* Add documentation for using gksu nautilus in lieu of "Root Terminal" in Tails 3 backup process (#2069).
* Update hardware requirements (#2207).

## 0.4.2

* Explicitly enables DAC override capability in Apache AppArmor profile (#2105)

The issues for this release were tracked in the 0.4.2 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.4.2.

## 0.4.1

* Fixes a bug in one of the Tails scripts used to set up the Desktop
icons for the SecureDrop interfaces (#2049)

The issues for this release were tracked in the 0.4.1 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.4.1.

## 0.4

The issues for this release were tracked in the 0.4 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.4.

This changelog shows major changes below. Please diff the tags to see the full list of changes.

### Deployment

* Enable optional HTTPS on the source interface (#1605).
* Standardize SecureDrop server installation on a single username (#1796).
* Add `securedrop-admin` script and update version of Ansible running in the workstation (#1146, #1885).
* Add validation of user-provided values during SecureDrop installation (#1663, #749, #1257).
* Removes `prod-specific.yml` configuration file (#1758).
* Allow an administrator to set a custom daily reboot time (#1515).
* Renames "document interface" to "journalist interface" (#1384, #1614).
* Adds "Email from" option to OSSEC (#894, #1220).
* Updated template for Firewall and adds instructions on how to use it (#1122, #1648)
* Bugfix: Re-enable logging on journalist interface (#1606).

### Developer Workflow

* Reconciles divergent master and develop branches (#1559).
* Increases unit test coverage to from 65% to 92%.
* Adds testinfra system configuration test suite (#1580).
* Removes unnecessary test wrappers (#1412).
* Major improvements to SecureDrop CI and testing flow including adding the staging environment to CI (#1067).

### Web App: Source

* Mask codename on source interface (#525).
* Replace confusing text on source interface landing bar (#1713).
* Refresh of source UI (#1536, #1895, #1604).
* Add metadata endpoint to source webapp for monitoring (#972).
* Begin using EFF wordlist (#1361).

### Web App: Journalist

* Feature: Add unread submission icon and select all unread button to submission view (#1353).
* Refresh of journalist UI (#1604).
* Adds minimum password length requirements for new journalist accounts (#980).
* Delete submissions that have had their sources deleted (#1188).
* Bugfix: Empty replies can no longer be sent to a source (#1715).
* Bugfix: Handle non hexadecimal digits for the 2FA secret (#1869).
* Bugfix: Handle token reuse for the 2FA secret on /admin/2fa (#1687).
* Bugfix: Handle attempts to make duplicate user accounts (#1693).
* Bugfix: Fix confusing UI on message/reply icons (#1258).

### Tails Environment

* Improve folder structure for SecureDrop documents (#383).
* Bugfix: Update provided KeePassX template to kdbx format (#1831).
* Bugfix: Ensure the filename is restored when uncompressing an archive (#1862).

### Documentation

* Adds "What makes SecureDrop unique?" guide (#469).
* Adds passphrase best practices guide (#1136).
* Adds SecureDrop promotion guide (#1134).
* Adds Administrator responsibilities guide (#1727).
* Other minor miscellaneous documentation improvements.

## 0.3.12

* Disables swap on Application Server via preinst script on
`securedrop-app-code` package hook. Swap was not previously disabled
permanently, so this automatic update will deactivate it, shred the swap
partition if it was in use, then disable swap entries in /etc/fstab.
A separate change in future release will enforce this configuration
via the Ansible config at install time (#1620)

## 0.3.11

* Instructs source to turn the Tor Browser security slider to High and to
get a new Tor Browser identity after logout. Turning the security slider to
high would disable the SVG icons so the icons and CSS on the source interface
are updated to display properly using this setting (#1567, #1480, #1522)
* Adds `/logout` route and button to base template (#1165)
* Removes the choice of number of codename words from the source interface
(#1521)
* Adds SourceClear integration (#1520)
* CSS fixes (#1186)
* Adds coveragerc

The issues for this release were tracked in the 0.3.11 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.3.11.

## 0.3.10

Creates new Debian package `securedrop-keyring` for managing the SecureDrop
Release Signing Key. Rotates the signing key currently in use by setting
a dependency on the other Debian packages, so that currently running deployments
will have their apt keyrings updated via automatic nightly updates.

* Installs securedrop-keyring package for managing apt signing key (#1416)

Admins must manually update the Release Signing Key on Admin Workstations.
See documentation on configuring the Admin Workstation.

The issues for this release were tracked in the 0.3.10 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.3.10.

## 0.3.9

Point release to fix some minor issues and update our Python dependencies.

* Fix Unicode support regression and implement better Unicode tests (#1370)
* Add OSSEC rule to ignore futile port scanning (#1374)
* Update Apache AppArmor profile to allow access to webfonts and to execute uname (#1332, #1373)
* Update Python dependencies of SD (#1379)
* Fix a regression in the new install script (#1397)

The issues for this release were tracked in the 0.3.9 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.3.9.

## 0.3.8

* Re-include the pycrypto Python module to address the regression in 0.3.7 (#1344)
* Switch to using bento boxes in Vagrantfile for more reproducible test environments
* Minor fixes to update_version.sh

The issues for this release were tracked in the 0.3.8 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.3.8

## 0.3.7

Point release to address some requests from SecureDrop administrators and
upgrade various components on the SecureDrop architecture.

* Improve backup and restore roles
* Ensure SecureDrop-specific Tails configuration works on Tails 2.x
* Document upgrading from Tails 1.x to Tails 2.x for current admins using 1.x
* Upgrade SecureDrop's Python dependencies

The issues for the release were tracked with the [0.3.7 milestone on
GitHub](https://github.com/freedomofpress/securedrop/milestones/0.3.7).

## 0.3.6

This is an emergency release to update the copy of the FPF code signing public
key in the repo because it expired on Oct 26. This fix is required for new
installs to succeed; otherwise, the installation will fail because apt's
package authentication fails if the corresponding key is expired.

## 0.3.5

The issues for this release were tracked with the 0.3.5 milestone on GitHub: https://github.com/freedomofpress/securedrop/milestones/0.3.5

* Use certificate verification instead of fingerprint verification by default for the OSSEC Postfix configuration (#1076)
* Fix apache2 service failing to start on Digital Ocean (#1078)
* Allow Apache to rotate its logs (#1074)
* Prevent reboots during cron-apt upgrade (#1071)
* Update documentation (#1107, #1112, #1113)
* Blacklist additional kernel modules used for wireless networking (#1116)

## 0.3.4

The issues for this release were tracked with the 0.3.4 milestone on GitHub: https://github.com/freedomofpress/securedrop/milestones/0.3.4

This release contains fixes for issues described in the most recent security audit by iSec. It also contains some improvements and updates to the documentation, and a fix for Tor hidden service directory permissions that caused new installs to fail.

### iSec audit fixes

* Fix ineffective SSH connection throttling (iSEC-15FTC-7, #1053)
* Remove debugging print statements that could leak sensitive information to the logs for the document interface (iSEC-15FTC-2, #1059)
* Harden default iptables policies (iSEC-15FTC-3, #1053)
* Don't check passwords or codenames that exceed a maximum length to prevent DoS via excessive scrypt computation (iSEC-15FTC-6, #1059)
* Remove unnecessary capabilities from the Apache AppArmor profile (iSEC-15FTC-9, #1058).
* Change postfix hostname to something generic to prevent fingerprinting via OSSEC email headers (iSEC-15FTC-10, #1057)

### Other changes

* Ensure correct permissions for Tor hidden service directories so new installs won't break (#1052)
* Clarify server setup steps in the install documentation (#1027, #1061)
* Clarify that Tor ATHS setup is now automatic and does not require manual changes (#1030)
* Explain that you can only download files to the "Tor Browser" folder on Tails as of Tails 1.3, due to the addition of AppArmor confinement for Tor Browser (#1036, #1062).
* Explain that you must use the Unsafe Browser to configure the network firewall because Tor Browser will be blocked from accessing LAN addresses starting in Tails 1.5 (#1050)
* Fix "gotcha" in network firewall configuration where pfSense guesses the wrong CIDR subnet (#1060)
* Update the upgrade docs to refer to the latest version of the 0.3.x release series instead of a specific version that would need to be updated every time (#1063)

## 0.3.3

The issues for this release were tracked with the 0.3.3 milestone on GitHub:
https://github.com/freedomofpress/securedrop/milestones/0.3.3.

* Remove unnecessary proxy command from Tails SSH aliases (#933)
* Make grsec reboot idempotent to avoid unnecessary reboots on new installs (#939)
* Make tmux the default shell on App and Monitor servers (#943)
* Fully tested migration procedures for 0.2.1 and 0.3pre to 0.3 (#944, #993)
* Ensure grub is not uninstalled in virtual machines (#945)
* CSS fixes (#948)
* Apache AppArmor profile should support TLS/SSL (#949)
* Fix: document interface no longer flas new submissions as unread (#969)
* Switch to NetworkManager for automatic ATHS setup on Admin Workstation (#1018)
* Upgrade Selenium in testing dependencies so functional tests work (#991)
* Clarify paths in install documentation (#1009)

## 0.3.2

* Fixes security vulnerability (severity=high) in access control on Document Interface (#974)

## 0.3.1

* Improved installation and setup documentation (#927, #907, #903, #900)
* Fixed PEP8 and other style issue (#926, #893, #884, #890, #885)
* Automatic torrc initialization in Tails via dotfiles persistence (#925)
* Fix bug in installing grsecurity kernel when using new Ubuntu 14.04.2 .iso (#919)
* Prevent sources from creating "empty" submissions (#918)
* Autoremove unused packages after automatic upgrade (#916)
* Remove the App Server (private) IP address from OSSEC alert email subject lines (#915)
* Handle custom header image as a conffile in the securedrop-app-code Debian package (#911)
* Upgrade path from 0.3pre (#908, #909)
* Remove offensive words from source and journalist word lists (#891, #901)

## 0.3

### Web App

This is a high-level overview of some of the more significant changes between SecureDrop 0.2 and 0.3. For the complete set of changes, diff the tags.

* Reduce JS dependencies to JQuery (stable) only
* Add functional tests, increase unit test coverage
* Rewrite database layer (db.py) using SQLAlchemy declarative ORM
* Automate dev. setup with Vagrant and integrate with Travis CI
* Store more info in db and less on filesystem
  * "flagged" sources
  * metadata for new UI features (starring, etc.)
  * metadata for simpler/more efficient views in journalist.py
* Do not set headers in the web app (handle by production config.)
* Add 2fac auth for journalist interface
* Allow OSSEC emails to be encrypted with admin GPG key
* Install app server, monitor server, Python dependencies, and custom configuration via Debian packages
* UI refresh on source and journalist interfaces
* New UX for journalists:
  * "quick filter" box for codenames
  * "download unread" link
  * star sources
  * multi-select actions for sources (delete, star, unstar) and submissions (download, delete)
  * more detailed source listings
* Normalize submission timestamps to that of the most recent submission
to minimize metadata that could be used for correlation
* Handle journalist authentication in the Document Interface instead of relying entirely on Authenticated Tor hidden services.
* Document Interface supports two-factor authentication via Google Authenticator or YubiKey
  * These logins are hardened in a manner similar to that of the `google-authenticator` PAM module: tokens may only be used once, logins are rate limited, etc.
  * If you are using TOTP, the window is expanded from 1 period to 3 in order to help the situation where the server and client's clocks are skewed
* Add Admin Interface so privileged "admin" users may add, edit, or delete other users on the Document Interface
* Requests are automatically encrypted with an ephemeral key as they are buffered onto disk to mitigate forensic attacks
* The haveged "high water" mark has been raised to maintain a higher average level of entropy on the system and minimize the appearance of the "flag for reply" flow
* Secure removal (via `srm`) of data has been moved to an async worker to prevent hanging the interface when deleting large files or collections
* New dedicated section of Source Interface for replies, instead of using flashed messages
* Change default codename length from 8 words to 7 words, maintains a sufficient security level while hopefully improving usability for sources
* Add recommendations for storing and memorizing the codename to the codename generation page
* Improve the quality of journalist designations generated by reducing the adjectives and nouns lists to a smaller subset of common words
* Use ntpd to continuously update the server time (especially important when using TOTP for two-factor authentication)
* Move Document Interface to port 80 so we don't have to keep remembering to type ":8080"
* We no longer ASCII-armor submissions when they are encrypted. This was unnecessary and bloated the size of the submissions, which is important to avoid because downloading large submissions over Tor is very slow.
* Flask now uses X-Send-File for downloads, which fixed some reported issues are large downloads not finishing or being corrupted.

### Environment

* Add egress host firewall rules
* Add google-authenticator Apache module and basic auth for access to
document interface
* Encrypt bodies of OSSEC email alerts (add postfix+procmail to monitor
server)
* Create apparmor profiles for chrooted interface Tor process
* Update interface apparmor profiles for changes to application code
* Change installation method to use Ansible playbook and deb packages
* Split securedrop repo into 3 separate repos for securedrop-specific code (the application, Python dependencies, and custom configuration) and the upstream packages that we maintain (OSSEC and the hardened grsecurity kernel for Ubuntu)
* Add variety of development and testing environments for developers and researchers to use with Vagrant
* Reduce OSSEC email alert noise through whitelisting errors that are reported by the default configuration but that we have investigated and determined to be safe to ignore
* Document a thoroughly tested network firewall configuration with pfSense
* Reboot the machine automatically every 24 hours to reduce the potential for plaintext to remain in memory
* Add KeePassX password database template and document its use for journalists and admins
* Add secure backup and recovery scripts
* Add migration scripts
* Major improvements to the installation and user documentation, including lots of detail, testing, and the addition of TOC

## 0.2.1

### Web App
* Fix for flagging errors
* Validate journalist messages
* Add logging using standard Python library.
* Add delete collection
* Replace bcrypt with scrypt
* Clear referer on external links

### Environment
* Set maximum request body size in CONFIG_OPTIONS
* Add security-related HTTP headers to Apache config
* Remove mysql database, replace w/ sqlite. Update sqlite apparmor profile.
* Add outbound iptables rule for source/document groups

## 0.2

* Various documentation improvements

### Web App
* Remove javascript dependency in source interface
* Add warning to source interface about using javascript (with Gritter)
* Update to pycrypto 2.6.1
* Validate filenames and codenames
* Remove unsafe characters from codenames, remove diceware words that are
not real words
* Rewrite source.py and journalist.py with Flask
* Add tests
* Flag sources for journalist reply to avoid DOS attack by generating
many GPG keys
* Allow journalists to delete documents with SRM
* Add bulk download to journalist interface
* Add MySQL-python and SQLAlchemy dependency, db.py to perform database
functions (ex: storing codenames)
* Remove option to have codenames with <7 words
* Use sqlite as default database
* Add support for theming
* bcrypt hash GPG passphrase for key stretching

### Environment
* Merge source and journalist servers into a single app server
* Add apparmor profiles
* Remove puppet, add base_install.sh script
* Create interface-install.sh script to set up chroot jails
* Add Ubuntu dev-setup script
* Backup Tor private keys
* Move config files into install scripts directory
* Change SOURCE_IP to APP_IP
* Set ownership and permissions for application code

## 0.1

* Renamed DeadDrop to SecureDrop
* Redesigned source and document web interface
* Wrote detailed documentation
* Improved installation process
* Wrote unit & integration tests
* Improved codename wordlist, based on Diceware
* Use bcrypt instead of SHA
* Removed VPN, replaced with authenticated Tor hidden service
* Freedom of the Press Foundation taking over project

DeadDrop was originally written by Aaron Swartz.
