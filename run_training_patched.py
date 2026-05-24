#!/usr/bin/env python3

import argparse

from waste_sort.cli import main

original_add_argument = argparse.ArgumentParser.add_argument


def patched_add_argument(self, *args, **kwargs):
    try:
        return original_add_argument(self, *args, **kwargs)
    except ValueError as e:
        if "badly formed help string" in str(e):
            kwargs["help"] = argparse.SUPPRESS
            return original_add_argument(self, *args, **kwargs)
        else:
            raise


argparse.ArgumentParser.add_argument = patched_add_argument


if __name__ == "__main__":
    main()
