# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0

import glob
import os.path

import grain.sources


INPUT_FIELDNAME = "input"
ANSWER_FIELDNAME = "answer"


class TestsDataSource(grain.sources.RandomAccessDataSource[dict]):
    """A data source backed by a directory of tests.

    The tests should have file patterns X.in and X.out, X being a number.
    """

    def __init__(self, directory, in_ext='.in', ans_ext='.out'):
        self._file_bases = _get_tests(directory, in_ext, ans_ext)
        self._in_ext = in_ext
        self._ans_ext = ans_ext

    def __len__(self) -> int:
        return len(self._file_bases)

    def __getitem__(self, index: int) -> tuple[str, str]:
        fn = self._file_bases[index]
        with open(fn + self._in_ext) as f:
            content1 = f.read()
        with open(fn + self._ans_ext) as f:
            content2 = f.read()
        return {INPUT_FIELDNAME: content1, ANSWER_FIELDNAME: content2}

    def __repr__(self) -> str:
        return str(('TestsDataSource', self._file_bases, self._in_ext,
                    self._ans_ext))


def _get_tests_basenames(directory, ext):
    files = glob.glob(os.path.join(directory, '*' + ext))
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)


def _get_tests(directory, in_ext, ans_ext):
    in_files = _get_tests_basenames(directory, in_ext)
    ans_files = _get_tests_basenames(directory, ans_ext)
    assert in_files == ans_files
    return [os.path.join(directory, f) for f in in_files]
