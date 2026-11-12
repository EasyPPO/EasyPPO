#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    std::string basePath = "";
    bool all_successful = true;

    std::cout << "Starting diagnostic read for .ans files in " << basePath << std::endl;

    for (int i = 1; i <= 10; ++i) {
        std::string filename = basePath + std::to_string(i) + ".ans";
        std::cout << "\n--- Reading file: " << filename << " ---" << std::endl;

        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "  [ERROR] Could not open the file. It may not exist." << std::endl;
            all_successful = false;
            continue;
        }

        // Check if the file is empty
        if (file.peek() == std::ifstream::traits_type::eof()) {
            std::cerr << "  [ERROR] File is completely empty." << std::endl;
            all_successful = false;
            file.close();
            continue;
        }

        std::vector<std::pair<int, int>> pairs;
        int r, c;
        while (file >> r >> c) {
            pairs.push_back({r, c});
        }

        // After the loop, check why it ended
        if (file.eof()) {
            // Good case: reached the end of the file.
            if (pairs.empty()) {
                 std::cerr << "  [WARNING] File opened but contained no valid integer pairs." << std::endl;
            } else {
                 std::cout << "  [SUCCESS] Successfully read " << pairs.size() << " pairs of integers." << std::endl;
            }
        } else if (file.fail()) {
            // Bad case: reading stopped due to a format error.
            std::cerr << "  [ERROR] Reading failed. The file may contain non-integer data or be malformed." << std::endl;
            all_successful = false;
        }

        file.close();
    }

    std::cout << "\n--- Summary ---" << std::endl;
    if (all_successful) {
        std::cout << "All checked .ans files were read successfully." << std::endl;
    } else {
        std::cout << "There were errors reading one or more .ans files. Please review the logs above." << std::endl;
    }

    return 0;
}