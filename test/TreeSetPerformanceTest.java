import java.util.Random;
import java.util.TreeSet;
import java.time.Duration;
import java.time.Instant;

public class TreeSetPerformanceTest {
    public static void main(String[] args) {
        TreeSet<Float> floatSet = new TreeSet<>();

        final int numElements = 10000000;

        float[] randomFloats = new float[numElements];

        Random random = new Random();

        for (int i = 0; i < numElements; ++i) {
            randomFloats[i] = random.nextFloat() * 100;
        }

        Instant startInsert = Instant.now();

        for (int i = 0; i < numElements; ++i) {
            floatSet.add(randomFloats[i]);
        }

        Instant endInsert = Instant.now();

        Duration elapsedInsert = Duration.between(startInsert, endInsert);
        System.out.println("Insert into set cost: " + elapsedInsert.toMillis() / 1000.0 + " s");

        Instant startErase = Instant.now();

        for (int i = 0; i < numElements; ++i) {
            floatSet.remove(randomFloats[i]);
        }

        Instant endErase = Instant.now();

        Duration elapsedErase = Duration.between(startErase, endErase);
        System.out.println("Erase set elements one by one cost: " + elapsedErase.toMillis() / 1000.0 + " s");
    }
}