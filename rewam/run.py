import runpy
import sys

from . import register


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m rewam_starvla.run MODULE [MODULE_ARGUMENTS...]")
        return
    module = sys.argv[1]
    if not module.startswith(("starVLA.", "deployment.")):
        raise ValueError("Expected a StarVLA training or deployment module.")
    register()
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
