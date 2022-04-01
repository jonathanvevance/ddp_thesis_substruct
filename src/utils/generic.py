"""Python file generic util functions."""

def nested2d_generator(list_A, list_B):
     """Product a stream of 2D coordinates."""
     for a in list_A:
        for b in list_B:
            yield a, b
