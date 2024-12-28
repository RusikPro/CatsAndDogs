import os
import argparse

################################################################################

def rename_images(folder_path, prefix):
    """
    Renames all files in 'folder_path' by adding 'prefix' to the front
    of each original filename (keeping the original name and extension intact).
    For example: example.jpg -> cat.example.jpg (if prefix="cat.").
    """
    files = sorted(os.listdir(folder_path))

    for filename in files:
        old_path = os.path.join(folder_path, filename)

        # Skip directories
        if os.path.isdir(old_path):
            continue

        # Build the new filename: prefix + original filename
        new_filename = f"{prefix}{filename}"
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_filename}'")

################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Rename images by adding a prefix (without numbering)."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the folder containing images."
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix to use when renaming (e.g. 'cat.' or 'dog.')."
    )

    args = parser.parse_args()

    # Validate folder
    if not os.path.isdir(args.folder):
        print(f"Error: The folder '{args.folder}' does not exist or is not a directory.")
        return

    rename_images(args.folder, args.prefix)

################################################################################

if __name__ == "__main__":
    main()
