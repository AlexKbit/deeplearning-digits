package com.alexkbit.deeplearning.digits;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.StringJoiner;
import java.util.stream.Stream;

import com.alexkbit.deeplearning.digits.util.ConvertUtil;

public class DataSetPrepare {

    public static String DATA_SET_FILE = "dataset/digits.txt";
    private static String DATA_DIRECTORY = "dataset/images50";

    public static void main(String[] args) throws IOException {
        FileWriter fw = new FileWriter(DATA_SET_FILE, true);
        BufferedWriter bw = new BufferedWriter(fw);
        PrintWriter out = new PrintWriter(bw);

        try (Stream<Path> paths = Files.walk(Paths.get(DATA_DIRECTORY))) {
            paths.filter(Files::isRegularFile)
                    .forEach(p -> createDataSet(p, out));
        }
        out.flush();
        out.close();
        System.out.println("End");
    }

    private static void createDataSet(Path file, PrintWriter out) {
        int[] tensor;
        try {
            tensor = ConvertUtil.convertToTensor(file);
            StringJoiner sj = new StringJoiner(",");
            Arrays.stream(tensor).forEach(v -> sj.add(Integer.toString(v)));
            String digitClass = file.getFileName().toFile().getName().substring(0,1);
            sj.add(digitClass);
            out.println(sj.toString());
            System.out.println("Create dataset from: " + file.getFileName() + " digit class: " + digitClass);
        } catch (IOException e) {
            System.out.println("Fail parse");
        }
    }
}
