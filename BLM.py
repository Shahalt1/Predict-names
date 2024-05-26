import torch
import matplotlib.pyplot as plt


class BigramLM:
    def __init__(self, text_path=None):
        self.text = None
        self.N = None
        self.stoi = None
        self.itos = None
        if text_path:
            self.upload_text(text_path)
            self.create_graph()

    def upload_text(self, path):
        with open(path, "r") as file:
            text = file.read().lower().split()
        stop_chars = "'"
        self.text = [word for word in text if stop_chars not in word]

    def create_graph(self):
        if not self.text:
            raise ValueError("Text not uploaded or provided")

        chars = sorted(set("".join(self.text)))
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}
        self.stoi["#"] = 0
        self.itos = {i: c for c, i in self.stoi.items()}

        size = len(self.stoi)
        self.N = torch.zeros((size, size), dtype=torch.float32)

        for word in self.text:
            word = ["#"] + list(word) + ["#"]
            for ch1, ch2 in zip(word, word[1:]):
                i = self.stoi[ch1]
                j = self.stoi[ch2]
                self.N[i, j] += 1

    def plot(self):
        if self.N is None:
            raise ValueError("Graph not created. Please run create_graph() first.")

        plt.figure(figsize=(16, 16))
        plt.imshow(self.N, cmap="Blues")
        size = len(self.stoi)
        for i in range(size):
            for j in range(size):
                chstr = self.itos[i] + self.itos[j]
                plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
                plt.text(
                    j, i, int(self.N[i, j].item()), ha="center", va="top", color="black"
                )
        plt.axis("off")
        plt.show()

    def create_names(self):
        if self.N is None:
            raise ValueError("Graph not created. Please run create_graph() first.")

        gen = torch.Generator().manual_seed(123123)
        output = []

        # Normalizing the bigram probabilities
        p = (self.N + 1).float()
        p /= p.sum(1, keepdim=True)

        for _ in range(20):
            name = []
            current_char = "#"
            while True:
                i = self.stoi[current_char]
                next_char_idx = torch.multinomial(
                    p[i], 1, replacement=True, generator=gen
                ).item()
                next_char = self.itos[next_char_idx]
                if next_char == "#":
                    break
                name.append(next_char)
                current_char = next_char
            output.append("".join(name))
        return output

    def loss(self):
        if self.N is None:
            raise ValueError("Graph not created. Please run create_graph() first.")

        lle = 0
        count = 0

        # Normalizing the bigram probabilities
        p = (self.N + 1).float()
        p /= p.sum(1, keepdim=True)

        for word in self.text:
            word = ["#"] + list(word) + ["#"]
            for ch1, ch2 in zip(word, word[1:]):
                i = self.stoi[ch1]
                j = self.stoi[ch2]
                lle += torch.log(p[i, j])
                count += 1

        return -lle / count
