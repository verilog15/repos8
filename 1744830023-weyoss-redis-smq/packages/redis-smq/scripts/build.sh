#
# Copyright (c)
# Weyoss <weyoss@protonmail.com>
# https://github.com/weyoss
#
# This source code is licensed under the MIT license found in the LICENSE file
# in the root directory of this source tree.
#

set -x
set -e

rm -rf dist

# esm
tsc -p ./tsconfig.json
cp -r src/common/redis-client/scripts/lua dist/esm/src/common/redis-client/scripts/

# cjs
tsc -p ./tsconfig.cjs.json
cat >dist/cjs/package.json <<!EOF
{
    "type": "commonjs"
}
!EOF
cp -r src/common/redis-client/scripts/lua dist/cjs/src/common/redis-client/scripts/
