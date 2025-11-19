import os
import sys
import json

# --- Configuration ---
# Set the name of the root directory you want to create and analyze.
# It will be created in the same location where you run this script.
ROOT_DIR_NAME = r"E:\ML\temps\CrediMax"

# New setting to exclude directories from being printed or traversed
EXCLUDE_FOLDERS = ['__pycache__', 'env', '.git', '.venv', '.hydra', '.venv', 'artifacts', 'mlruns']

# NEW: Define file extensions for which content should be read
SOURCE_CODE_EXTENSIONS = ('.py', '.yaml', '.toml', '.js', '.html', '.css', '.md', '.txt', '.rst')
# Added common text files (.md, .txt, .rst) to ensure documentation is included.

# Define the structure to be created
SAMPLE_STRUCTURE = {
    "docs": [
        "README.md",
        "LICENSE.txt",
        {"api": ["__init__.py", "users.rst"]},
    ],
    "src": [
        {"models": ["user.py", "post.py"]},
        {"views": ["__init__.py", "base_view.py"]},
        "main.py",
        "config.py",
    ],
    "tests": [
        "test_main.py",
        "test_config.py",
    ],
    ".gitignore": None
}

def create_structure(base_path, structure):
    """
    Recursively creates the directory and file structure defined in the dictionary.
    """
    if not isinstance(structure, dict):
        return

    for name, content in structure.items():
        current_path = os.path.join(base_path, name)
        
        if isinstance(content, list):
            # It's a directory containing files or other directories
            os.makedirs(current_path, exist_ok=True)
            for item in content:
                if isinstance(item, str):
                    # It's a file
                    with open(os.path.join(current_path, item), 'w') as f:
                        f.write(f"# This is {item} inside {name}\n")
                elif isinstance(item, dict):
                    # It's a nested directory
                    create_structure(current_path, item)
        elif content is None:
            # It's a standalone file (e.g., .gitignore at the root)
            with open(current_path, 'w') as f:
                f.write(f"# This is the {name} file\n")
        else:
            # Just in case of simple file list (should be handled by list iteration)
            os.makedirs(current_path, exist_ok=True)


# --- UPDATED FUNCTION FOR JSON CREATION ---
def create_json_with_content(startpath, output_filename="project_contents.json", exclude_dirs=None):
    """
    Traverses the directory structure and creates a JSON file.
    - If the file has a source code extension, the value is the file content.
    - Otherwise, the value is the file path itself.
    """
    if exclude_dirs is None:
        exclude_dirs = []
        
    project_data = {}
    
    print(f"\n--- Generating JSON file: {output_filename} ---")

    for root, dirs, files in os.walk(startpath):
        # Apply the exclusion logic
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file_name in files:
            full_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(full_path, startpath)
            
            # Get the file extension for filtering
            _, file_extension = os.path.splitext(file_name)
            
            # Check if we should read the content
            if file_extension in SOURCE_CODE_EXTENSIONS:
                try:
                    # Read the file content
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    project_data[relative_path] = content
                    print(f"  > Read content for: {relative_path}")
                    
                except UnicodeDecodeError:
                    # Handle files that claim to be text but aren't
                    project_data[relative_path] = relative_path
                    print(f"  > Skipping non-text/binary file, value is path: {relative_path}")
                except Exception as e:
                    project_data[relative_path] = relative_path
                    print(f"  > Error reading file {relative_path}. Value set to path. Error: {e}")
            else:
                # For non-source/non-text files, the value is the path itself
                project_data[relative_path] = relative_path
                print(f"  > Skipping file type, value is path: {relative_path}")

    # Write the resulting dictionary to a JSON file
    try:
        output_path = os.path.join(startpath, output_filename)
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(project_data, json_file, indent=4) 
        print(f"Successfully created JSON file at: {output_path}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")
# ------------------------------------


def simple_directory_tree_print(startpath, exclude_dirs=None):
    """
    A robust and simple directory tree printer using os.walk.
    It can now exclude directories specified in the exclude_dirs list.
    """
    if exclude_dirs is None:
        exclude_dirs = []
        
    print(f"\n--- Simple Directory Tree Print for: {os.path.abspath(startpath)} (Excluding: {', '.join(exclude_dirs)}) ---")
    
    # Define the visual markers
    INDENT = '    ' # 4 spaces for indentation
    
    # Print the root directory
    print(f"\033[1m{os.path.basename(startpath)}/\033[0m")
    
    for root, dirs, files in os.walk(startpath):
        
        # --- EXCLUSION LOGIC ---
        # Modify the dirs list in place to prevent os.walk from descending into 
        # excluded directories AND to prevent printing them in the current level.
        dirs[:] = [d for d in dirs if d not in exclude_dirs] 
        # -----------------------
        
        if root == startpath:
            # We already printed the root, now just sort contents
            dirs.sort()
            files.sort()
            continue
            
        # Determine the relative path and level
        relpath = os.path.relpath(root, startpath)
        parts = relpath.split(os.sep)
        level = len(parts)

        # Calculate indentation (e.g., level 1 is 4 spaces, level 2 is 8 spaces)
        current_indent = INDENT * (level - 1)
        
        # Print the directory itself
        # parts[-1] is the current directory's name
        print(f"{current_indent}├── \033[94m{parts[-1]}/\033[0m")

        # Sort files and directories for consistent output
        dirs.sort()
        files.sort()
        
        # Calculate indentation for the *contents* of the current directory
        content_indent = INDENT * level
        
        # Print Files
        for i, f in enumerate(files):
            # Determine connector: Last item uses '└──', others use '├──'
            connector = '└── ' if i == len(files) - 1 else '├── '
            print(f"{content_indent}{connector}{f}")


if __name__ == "__main__":
    # The root directory for the sample structure
    root_path = ROOT_DIR_NAME 

    # Helper for creating excluded folders and a dummy file inside them for demonstration
    def create_excluded_folders(root):
        for folder in EXCLUDE_FOLDERS:
             # Ensure the folder is created inside the root path
             excluded_path = os.path.join(root, folder)
             os.makedirs(excluded_path, exist_ok=True)
             with open(os.path.join(excluded_path, 'dummy_file.txt'), 'w') as f:
                 f.write("This file should not appear in the tree.")

    if not os.path.exists(root_path):
        print(f"Creating sample directory structure in: {root_path}")
        os.makedirs(root_path, exist_ok=True)
        # 2. Create the sample structure
        create_structure(root_path, SAMPLE_STRUCTURE)
        print("Sample structure created successfully.")
        
        # Create the excluded folders for demonstration purposes
        create_excluded_folders(root_path)
    else:
        print(f"Warning: Directory '{ROOT_DIR_NAME}' already exists. Skipping main structure creation.")
        create_excluded_folders(root_path) # Ensure excluded folders exist for demonstration
        print("Ensured excluded folders exist for demonstration.")


    # 3. Print the directory tree
    # Call the robust printing function, passing the exclusion list
    simple_directory_tree_print(root_path, EXCLUDE_FOLDERS)

    # 4. Generate the JSON file with file contents (NEW STEP)
    # The JSON file will be placed in the root_path directory.
    create_json_with_content(root_path, "project_summary.json", EXCLUDE_FOLDERS)

    print("\n--- Cleanup Note ---")
    print(f"To remove the created sample directory, run: rm -rf {ROOT_DIR_NAME}")