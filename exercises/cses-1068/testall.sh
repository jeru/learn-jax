#! /bin/bash
set -e

run_test() {
  local opt="$1"
  local fin="$2"
  local fout="$(echo "$fin" | sed 's/in$/out/')"
  echo
  echo "==================================================="
  echo "Testing: $opt against $fin"
  echo
  rm -f tmp.out
  time uv run solve.py "$opt" < "$fin" > tmp.out
  diff -q -w tmp.out "$fout" && echo "OK" || { echo "FAILED!!!!!"; exit 1; }
}

for fin in $(ls testdata/*.in); do
  run_test opt1 $fin
  run_test opt2 $fin
done

