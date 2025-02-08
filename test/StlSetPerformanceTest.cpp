#include <iostream>
#include <set>
#include <random>
#include <chrono>
#include <vector>

int main() {
    std::set<float> floatSet;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    const int numElements = 10000000;
    std::vector<float> randomFloats(numElements);
    for (int i = 0; i < numElements; ++i) {
        randomFloats[i] = dis(gen);
    }

    auto startInsert = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numElements; ++i) {
        floatSet.insert(randomFloats[i]);
    }
    auto endInsert = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsedInsert = endInsert - startInsert;
    std::cout << "Insert into set cost: " << elapsedInsert.count() << " s" << std::endl;

    auto startErase = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numElements; ++i) {
        floatSet.erase(randomFloats[i]);
    }
    auto endErase = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsedErase = endErase - startErase;
    std::cout << "Erase set elements one by one cost: " << elapsedErase.count() << " s" << std::endl;

    return 0;
}