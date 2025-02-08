#include <iostream>
#include <map>
#include <random>
#include <chrono>
#include <vector>

int main() {
    // 创建一个map来存储浮点数
    std::map<float, int> floatMap;

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    // 生成1000000个随机浮点数并存储到静态数组中
    const int numElements = 1000000;
    std::vector<float> randomFloats(numElements);
    for (int i = 0; i < numElements; ++i) {
        randomFloats[i] = dis(gen);
    }

    // 开始计时插入操作
    auto startInsert = std::chrono::high_resolution_clock::now();

    // 逐一插入到map中
    for (int i = 0; i < numElements; ++i) {
        floatMap[randomFloats[i]] = i;  // 使用i作为value，只是为了填充map
    }


    auto endInsert = std::chrono::high_resolution_clock::now();

    // 计算插入操作的耗时
    std::chrono::duration<double> elapsedInsert = endInsert - startInsert;
    std::cout << "Insert into map cost: " << elapsedInsert.count() << " s" << std::endl;

    // 开始计时逐一删除操作
    auto startErase = std::chrono::high_resolution_clock::now();

    // 逐一删除map中的元素（按照插入的顺序）
    for (int i = 0; i < numElements; ++i) {
        floatMap.erase(randomFloats[i]);  // 根据静态数组中的值逐一删除
    }

    // 结束计时逐一删除操作
    auto endErase = std::chrono::high_resolution_clock::now();

    // 计算逐一删除操作的耗时
    std::chrono::duration<double> elapsedErase = endErase - startErase;
    std::cout << "Erase map elements one by one cost: " << elapsedErase.count() << " s" << std::endl;

    return 0;
}