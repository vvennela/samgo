"""BrickGPT's prompts, copied verbatim from the released code (src/brickgpt/models/brickgpt.py) so the
replication trains and decodes on exactly their instruction format."""

SYSTEM = "You are a helpful assistant."


def instruction(caption: str) -> str:
    return ('Create a LEGO model of the input. Format your response as a list of bricks: '
            '<brick dimensions> <brick position>, where the brick position is (x,y,z).\n'
            'Allowed brick dimensions are 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.\n'
            'All bricks are 1 unit tall.\n\n'
            '### Input:\n'
            f'{caption}')


def messages(caption: str, bricks: str | None = None) -> list[dict]:
    out = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction(caption)}]
    if bricks is not None:
        out.append({"role": "assistant", "content": bricks})
    return out
