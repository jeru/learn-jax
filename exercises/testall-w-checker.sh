#! /bin/bash
set -e

run_test() {
  local fin="$1"
  local fout="$(echo "$fin" | sed 's/in$/out/')"
  echo
  echo "==================================================="
  echo "Testing: $fin"
  echo
  rm -f tmp.out
  time uv run solve.py < "$fin" > tmp.out
  uv run checker.py "$fin" tmp.out "$fout"
}

for fin in $(ls -v testdata/*.in); do
  run_test $fin
done

