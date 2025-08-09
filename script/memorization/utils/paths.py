import os

# Utilities for getting inputs and outputs directories

class Paths:
    def __init__(self, root_dir, in_path, out_path, model, book):
        """
        root_dir: base directory that contains both input and output folders
        in_path: input folder name or relative path inside root_dir (e.g., "data")
        out_path: output folder name or relative path inside root_dir (e.g., "outputs")
        model: model name string
        book: book filename (with extension)
        """
        self.root_dir = root_dir
        self.input_root = os.path.join(root_dir, in_path)
        self.output_root = os.path.join(root_dir, out_path)
        self.model = model
        self.book = book

    @property
    def book_name(self):
        # Filename without extension, e.g. "book1.txt" -> "book1"
        return os.path.splitext(self.book)[0]

    # Full input path to the book file
    @property
    def input_book_path(self):
        return os.path.join(self.input_root, self.book)

    # Output directories by type, nested by model and book_name
    def manifest_dir(self):
        path = os.path.join(self.output_root, "manifests", self.model, self.book_name)
        os.makedirs(path, exist_ok=True)
        return path

    def probs_dir(self):
        path = os.path.join(self.output_root, "probs", self.model, self.book_name)
        os.makedirs(path, exist_ok=True)
        return path

    def metadata_dir(self):
        path = os.path.join(self.output_root, "metadata", self.model, self.book_name)
        os.makedirs(path, exist_ok=True)
        return path

    def runtime_dir(self):
        path = os.path.join(self.output_root, "runtime", self.model, self.book_name)
        os.makedirs(path, exist_ok=True)
        return path

    # Manifest file path always named manifest.json inside manifest_dir
    def manifest_path(self):
        return os.path.join(self.manifest_dir(), "manifest.json")

    def metadata_batch_path(self, batch_idx):
        filename = f"batch-{batch_idx}-inference-metadata.pkl"
        return os.path.join(self.metadata_dir(), filename)

    def runtime_path(self):
        return os.path.join(self.runtime_dir(), "runtime.json")

    # Config directory inside probs for given config params
    def config_dir(self, temperature, k, suffix_start):
        config_name = self.get_config_name(temperature, k, suffix_start)
        path = os.path.join(self.probs_dir(), config_name)
        os.makedirs(path, exist_ok=True)
        return path

    # Path to a batch file inside a specific config directory
    def probs_path(self, temperature, k, suffix_start, batch_idx):
        cfg_dir = self.config_dir(temperature, k, suffix_start)
        return os.path.join(cfg_dir, f"batch-{batch_idx}-results.json")

    def probs_path_from_config(self, config_item, batch_idx):
        temperature = config_item.temperature
        k = config_item.k
        suffix_start = config_item.suffix_start_index
        return self.probs_path(temperature, k, suffix_start, batch_idx)

    # Helpers for config naming
    @staticmethod
    def get_config_name(temperature, k, suffix_start):
        temp_str = Paths.format_temp_str(temperature)
        return f"config-temp_{temp_str}-k_{k}-suffix-start_{suffix_start}"

    @staticmethod
    def format_temp_str(value: float) -> str:
        if value == int(value):
            return f"{int(value)}.0"
        elif value < 1:
            return f"0{value:.10g}"
        else:
            return f"{value:.10g}"
