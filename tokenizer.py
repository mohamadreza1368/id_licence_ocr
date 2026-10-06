import config

class CTCLabelConverter:
    """
    Convert between text-strings and integer labels for CTC loss.
    Index 0 is reserved for CTC Blank token [blank].
    """
    def __init__(self, character_list=config.ALL_CHARS):
        # 0 is blank
        self.blank_idx = 0
        self.character = ["[blank]"] + list(dict.fromkeys(character_list))
        self.char2idx = {char: i for i, char in enumerate(self.character)}
        self.idx2char = {i: char for i, char in enumerate(self.character)}
        self.num_classes = len(self.character)

    def normalize(self, text: str) -> str:
        """
        Normalize Persian/Arabic characters:
        e.g. arabic kaf/yeh to standard persian, unify spaces.
        """
        text = text.replace("ي", "ی").replace("ك", "ک").replace("ئ", "ی")
        # Remove unwanted control characters / zero-width non-joiners if needed
        text = text.replace("\u200c", " ")  # replace zwnj with space or remove
        return text.strip()

    def encode(self, text_list):
        """
        Encode a list of text strings into a 1D target tensor and target lengths.
        """
        targets = []
        lengths = []
        for text in text_list:
            text = self.normalize(text)
            encoded = [self.char2idx[char] for char in text if char in self.char2idx]
            targets.extend(encoded)
            lengths.append(len(encoded))
        return targets, lengths

    def decode_greedy(self, preds_indices):
        """
        Greedy CTC decoder:
        Input: preds_indices of shape [T, B] or [B, T]
        Collapse duplicate characters and remove blank (0).
        """
        results = []
        for seq in preds_indices:
            decoded_chars = []
            prev_idx = -1
            for idx in seq:
                idx = int(idx)
                if idx != self.blank_idx and idx != prev_idx:
                    decoded_chars.append(self.idx2char.get(idx, ""))
                prev_idx = idx
            results.append("".join(decoded_chars))
        return results
