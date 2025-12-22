from baseline.dummy import run_dummy_baseline
from rules_baseline import rules_model
from rules_baseline.rules_model import run_rules_baseline


def main():
    print("main started")
    print("########### start dummy model ###########")
    run_dummy_baseline()
    print("########### end dummy model ###########")

    print("########### start rules model ###########")
    run_rules_baseline()
    print("########### end rules model ###########")
    print("main finished")

if __name__ == "__main__":
    main()
