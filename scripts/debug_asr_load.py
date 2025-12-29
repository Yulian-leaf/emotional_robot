import sys
import traceback


def main() -> int:
    print("python", sys.version)
    try:
        from transformers import pipeline
    except Exception as e:
        print("IMPORT transformers FAILED:", repr(e))
        traceback.print_exc()
        return 2

    model_id = "openai/whisper-tiny"
    print("loading ASR pipeline:", model_id)
    try:
        asr = pipeline("automatic-speech-recognition", model=model_id)
        print("ASR LOADED:", type(asr))
    except Exception as e:
        print("ASR LOAD FAILED:", repr(e))
        traceback.print_exc()
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
