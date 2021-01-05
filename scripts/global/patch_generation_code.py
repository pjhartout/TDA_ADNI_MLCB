#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""patch_generation_code.py

This piece of code was provided by Sarah to generate the patch labels.

"""

import numpy as np


def main():
    ids = np.zeros((6, 6, 6))
    innerPatch = []
    for patch in range(0, 216):
        patchid2 = (
            patch
            - np.floor(patch / 36) * 36
            - np.floor((patch - np.floor(patch / 36) * 36) / 6) * 6
        )
        patchid1 = np.floor((patch - np.floor(patch / 36) * 36) / 6)
        patchid0 = np.floor(patch / 36)
        ids[int(patchid0), int(patchid1), int(patchid2)] = patch
        print(
            "Patch "
            + str(patch)
            + ": "
            + str(patchid0)
            + " "
            + str(patchid1)
            + " "
            + str(patchid2)
        )


if __name__ == "__main__":
    main()
