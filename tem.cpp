#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

void createFile(const std::string &path, const std::string &content) {
    std::ofstream file(path);
    if (file.is_open()) {
        file << content;
        file.close();
        std::cout << "Created file: " << path << std::endl;
    } else {
        std::cerr << "Failed to create file: " << path << std::endl;
    }
}

void setupProjectStructure(const std::string &rootFolder) {
    // Define project structure
    std::vector<std::string> directories = {
        rootFolder,
        rootFolder + "/data",
        rootFolder + "/models",
        rootFolder + "/src",
        rootFolder + "/logs",
        rootFolder + "/configs"
    };

    // Create directories
    for (const auto &dir : directories) {
        fs::create_directories(dir);
        std::cout << "Created directory: " << dir << std::endl;
    }

    // Create initial files
    createFile(rootFolder + "/README.md", "# Project: MLOps Template\n\nThis is a template for an end-to-end MLOps project.");
    createFile(rootFolder + "/requirements.txt", "# Python dependencies\nnumpy\npandas\nscikit-learn\n");
    createFile(rootFolder + "/configs/config.json", R"({
    \"dataset_path\": \"data/dataset.csv\",
    \"model_path\": \"models/model.pkl\",
    \"log_path\": \"logs/app.log\"
})");
    createFile(rootFolder + "/src/main.cpp", "#include <iostream>\n\nint main() {\n    std::cout << \"MLOps Project Setup Complete!\" << std::endl;\n    return 0;\n}");
    createFile(rootFolder + "/data/.gitkeep", ""); // Empty file to keep the directory in version control
    createFile(rootFolder + "/models/.gitkeep", "");
    createFile(rootFolder + "/logs/.gitkeep", "");
}

int main() {
    std::string projectName;
    std::cout << "Enter the project name: ";
    std::cin >> projectName;

    if (projectName.empty()) {
        std::cerr << "Project name cannot be empty!" << std::endl;
        return 1;
    }

    setupProjectStructure(projectName);

    std::cout << "MLOps project setup complete in folder: " << projectName << std::endl;
    return 0;
}
