def remove_newlines(input_string: str) -> str:
    return input_string.replace("\n", "")

if __name__ == "__main__":
    input = """
a bitch, a nerd, a psycho. I said good
thing I wasn't her, or I'd have taken a shit on
the ground. I'm not pretty. I'm not smooth. I'm not
bright. I'm not fierce. I'm not gorgeous. I'm not
normal. I'm not normal. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm
not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'm not beautiful. I'
"""
    res = remove_newlines(input)
    print(res)
