// Catalano Genetic Library
// The Catalano Framework
//
// Copyright © Diego Catalano, 2012-2016
// diego.catalano at live.com
//
package Catalano.Genetic.Optimization;

import Catalano.Core.DoubleRange;
import Catalano.Math.Matrix;
import Catalano.Math.Tools;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Firefly Optimization (FA).
 * 
 * The firefly algorithm is a metaheuristic proposed by Xin-She Yang and inspired
 * by the flashing behaviour of fireflies.
 * 
 * @author Diego Catalano
 */
public class FireflyOptimization implements IOptimization{
    
    private int population;
    private int generations;
    
    private double alpha = 0.2;
    private double beta0 = 2;
    private double gamma = 1;
    private double alphaDamp = 0.98;
    private double delta = 0.05;
    private double dmax;
    
    private double[] best;
    private double minError;
    private int nEval;
    
    /**
     * Get mutation coefficient.
     * @return Alpha value.
     */
    public double getMutationCoefficient(){
        return alpha;
    }
    
    /**
     * Set mutation coefficient.
     * @param alpha Alpha value.
     */
    public void setMutationCoefficient(double alpha){
        this.alpha = alpha;
    }
    
    /**
     * Get attraction coefficient
     * @return Beta0 value.
     */
    public double getAttractionCoefficient(){
        return beta0;
    }
    
    /**
     * Set Attraction Coefficient.
     * @param beta0 Beta0 value.
     */
    public void setAttractionCoefficient(double beta0){
        this.beta0 = beta0;
    }
    
    /**
     * Get light absorption coefficient.
     * @return Gamma value.
     */
    public double getLightAbsorptionCoefficient(){
        return gamma;
    }
    
    /**
     * Set light absorption coefficient.
     * @param gamma Gamma value.
     */
    public void setLightAbsorptionCoefficient(double gamma){
        this.gamma = gamma;
    }
    
    /**
     * Get damping ratio
     * @return Damping value.
     */
    public double getDampingRatio(){
        return alphaDamp;
    }
    
    /**
     * Set damping ratio.
     * @param damp Damping value.
     */
    public void setDampingRatio(double damp){
        this.alphaDamp = damp;
    }
    
    /**
     * Get uniform mutation ratio.
     * @return Delta value.
     */
    public double getUniformMutationRatio(){
        return delta;
    }
    
    /**
     * Set uniform mutation ratio.
     * @param delta Delta value.
     */
    public void setUniformMutationRatio(double delta){
        this.delta = delta;
    }

    @Override
    public int getNumberOfEvaluations() {
        return nEval;
    }
    
    @Override
    public double getError() {
        return minError;
    }

    /**
     * Initializes a new instance of the FireflyOptimization class.
     */
    public FireflyOptimization(){
        this(25, 1000);
    }
    
    /**
     * Initializes a new instance of the FireflyOptimization class.
     * @param population Number of population.
     * @param generations Number of generations.
     */
    public FireflyOptimization(int population, int generations){
        this(population, generations, 0.2, 2, 1);
    }
    
    /**
     * Initializes a new instance of the FireflyOptimization class.
     * @param population Number of population.
     * @param generations Number of generations.
     * @param alpha Mutation coefficient.
     * @param beta0 Attraction Coefficient Base Value.
     * @param gamma Light Absorption Coefficient.
     */
    public FireflyOptimization(int population, int generations, double alpha, double beta0, double gamma){
        this(population, generations, alpha, beta0, gamma, 0.98, 0.05);
    }
    
    /**
     * Initializes a new instance of the FireflyOptimization class.
     * @param population Number of population.
     * @param generations Number of generations.
     * @param alpha Mutation coefficient.
     * @param beta0 Attraction Coefficient Base Value.
     * @param gamma Light Absorption Coefficient.
     * @param alphaDamp Mutation Coefficient Damping Ratio.
     * @param delta Uniform Mutation Range Base Value.
     */
    public FireflyOptimization(int population, int generations, double alpha, double beta0, double gamma, double alphaDamp, double delta){
        this.population = population;
        this.generations = generations;
        this.beta0 = beta0;
        this.gamma = gamma;
        this.alphaDamp = alphaDamp;
        this.delta = delta;
    }

    @Override
    public double[] Compute(ISingleObjectiveFunction function, List<DoubleRange> boundConstraint) {
        
        double damp = alpha;
        minError = Double.MAX_VALUE;
        nEval = 0;
        
        Random rand = new Random();
        
        //Create initial population
        List<Firefly> pop = new ArrayList<Firefly>(population);
        
        for (int i = 0; i < population; i++) {
            
            double[] loc = Matrix.UniformRandom(boundConstraint);
            double fit = function.Compute(loc);
            pop.add(new Firefly(loc, fit));
            
            if(fit < minError){
                minError = fit;
                best = Arrays.copyOf(pop.get(i).location, boundConstraint.size());
            }
        }
        
        //Calculate dmax
        double[] min = new double[boundConstraint.size()];
        double[] max = new double[boundConstraint.size()];
        for (int i = 0; i < min.length; i++) {
            DoubleRange range = boundConstraint.get(i);
            min[i] = range.getMin();
            max[i] = range.getMax();
        }
        dmax = Matrix.Norm2(Matrix.Subtract(min, max));
        
        //Firefly algorithm
        for (int g = 0; g < generations; g++) {
            
            //Initialize a new population
            List<Firefly> newPop = new ArrayList<Firefly>(population);
            for (int i = 0; i < population; i++) {
                newPop.add(new Firefly(null, Double.MAX_VALUE));
            }
            
            //Compute
            for (int i = 0; i < population; i++) {
                for (int j = 0; j < population; j++) {
                    double rij = Matrix.Norm2(Matrix.Subtract(pop.get(i).location, pop.get(j).location)) / dmax;
                    double beta = beta0 * Math.exp(-gamma * (rij*rij));
                    double[] e = Matrix.Multiply(Matrix.UniformRandom(-1, 1, boundConstraint.size()), delta);

                    //New solution
                    double[] newsol = new double[boundConstraint.size()];
                    for (int k = 0; k < newsol.length; k++) {
                        Firefly a = pop.get(i);
                        Firefly b = pop.get(j);
                        newsol[k] = a.location[k] + beta * rand.nextDouble() * (b.location[k] - a.location[k]) + damp * e[k];
                        newsol[k] = Tools.Clamp(newsol[k], boundConstraint.get(j));
                    }

                    double newfit = function.Compute(newsol);
                    nEval++;
                    if(newfit <= newPop.get(i).fitness){
                        newPop.set(i, new Firefly(newsol, newfit));
                        if(newfit < minError){
                            minError = newfit;
                            best = Arrays.copyOf(newPop.get(i).location, boundConstraint.size());
                        }
                    }
                }
            }
            
            //Merge population
            pop.addAll(newPop);
            
            //Sort
            pop.sort(new Comparator<Firefly>() {

                @Override
                public int compare(Firefly o1, Firefly o2) {
                    return Double.compare(o1.fitness, o2.fitness);
                }
            });
            
            //Truncate
            pop = pop.subList(0, population);
        }
        damp = damp*alphaDamp;
        
        return best;
        
    }
    
    class Firefly{
        
        public double[] location;
        public double fitness;
        
        public Firefly(double[] location){
            this(location, Double.MAX_VALUE);
        }

        public Firefly(double[] location, double fitness) {
            this.location = location;
            this.fitness = fitness;
        }
        
    }
}
