def main() -> None:
    raise SystemExit(
        "The separate Joint stage was retired by Context V2. "
        "Run scripts/train_context.py; it already trains the Context field and "
        "AE decoder end to end."
    )


if __name__ == "__main__":
    main()
