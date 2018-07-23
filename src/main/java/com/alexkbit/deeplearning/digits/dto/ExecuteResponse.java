package com.alexkbit.deeplearning.digits.dto;

import java.util.ArrayList;

public class ExecuteResponse {
    private ArrayList<Double> probabilities = new ArrayList<>();
    private int answer;
    private double eps;

    public void addProbability(Double p) {
        probabilities.add(p);
    }

    public void setAnswer(int answer) {
        this.answer = answer;
    }

    public ArrayList<Double> getProbabilities() {
        return probabilities;
    }

    public int getAnswer() {
        return answer;
    }

    public double getEps() {
        return eps;
    }

    public void setEps(double eps) {
        this.eps = eps;
    }
}
