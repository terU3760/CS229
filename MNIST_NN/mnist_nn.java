import java.io.File; 
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

public class mnist_nn implements LearningEventListener{

	public final int N_DIGIT = 10, IMG_SIZE = 196, TRAIN_SIZE = 800, TEST_SIZE = 50, NN = 25, MAX_ITER = 10000;
	public final double A_TOL = 0.02, ALPHA = 1e-4, M = .875;  //RESERVE 50 random examples of each category for testing (cannot be used for training)
	public final String DELIMITER = ",";
	private MultiLayerPerceptron MLP;  

	public void run(final String FILENAME){
		int sz, min_sz = Integer.MAX_VALUE;
		ArrayList<ArrayList<double[]>> data = new ArrayList<ArrayList<double[]>>(N_DIGIT);
		DataSet train = new DataSet(IMG_SIZE + 1, 10), test = new DataSet(IMG_SIZE + 1, 10);
		for (int i = 0; i < N_DIGIT; ++i){
			data.add(new ArrayList<double[]>());
		}
		try{
			Scanner sc = new Scanner(new File(FILENAME));
			while (sc.hasNextLine()){
				Scanner l = new Scanner(sc.nextLine()).useDelimiter(DELIMITER);
				double[] digit = new double[IMG_SIZE + 1]; 
				for (int i = 0; i < IMG_SIZE; ++i){
					digit[i] = l.nextInt();   //converting from 0-7 scale to 0-255 scale
				}
				digit[IMG_SIZE] = -7.0;
				data.get(l.nextInt()).add(digit);
			}
			for (int i = 0; i < N_DIGIT; ++i){
				if ((sz = data.get(i).size()) < min_sz){
					if ((min_sz = sz) < TEST_SIZE){    //emit an error if size of a class is less than TEST_SIZE
						break;
					}
				}
				Collections.shuffle(data.get(i), new Random(System.nanoTime()));
			}
			if (min_sz >= TRAIN_SIZE + TEST_SIZE){ 
				for (int i = 0; i < N_DIGIT; ++i){
					double[] desiredOutput = new double[N_DIGIT];
					for (int j = 0; j < i; ++j){
						desiredOutput[j] = 0.0;
					}
					desiredOutput[i] = 1.0;
					for (int j = i + 1; j < N_DIGIT; ++j){
						desiredOutput[j] = 0.0;
					}
					for (int j = data.get(i).size() - TEST_SIZE; j < data.get(i).size(); ++j){
						test.addRow(new DataSetRow(data.get(i).get(j), desiredOutput));
					}
				}
				for (int j = 0; j < TRAIN_SIZE; ++j){
					for (int i = 0; i < N_DIGIT; ++i){ 
						double[] desiredOutput = new double[N_DIGIT];
						for (int k = 0; k < i; ++k){
							desiredOutput[k] = 0.0;
						}
						desiredOutput[i] = 1.0;
						for (int k = i + 1; k < N_DIGIT; ++k){
							desiredOutput[k] = 0.0;
						}
						train.addRow(new DataSetRow(data.get(i).get(j), desiredOutput));
					}
				}
			}else{
				System.err.println("Error: insufficient training data");
			}
			MLP = new MultiLayerPerceptron(TransferFunctionType.TANH, IMG_SIZE + 1, NN, NN, NN, N_DIGIT);
			MomentumBackpropagation LR = new MomentumBackpropagation();
			LR.setLearningRate(ALPHA);
			LR.setMomentum(M);
			LR.setMaxError(A_TOL);
			LR.setMaxIterations(MAX_ITER);
			LR.setBatchMode(false);
			LR.addListener(this);
			MLP.setLearningRule(LR);
			System.out.println("Training neural network...");
			MLP.learn(train);
			MLP.save("mnist_result_max_iter_10000.nnet");
			System.out.println("Testing trained neural network");
			testNN(MLP, test);
		}catch (Exception e){
			System.err.println("Error: " + e.getMessage());
		}
	}
	
	public void testNN(NeuralNetwork neuralNet, DataSet testSet) {
		int score = 0;
		for(DataSetRow testSetRow : testSet.getRows()) {
			neuralNet.setInput(testSetRow.getInput());
			neuralNet.calculate();
			int result = 0, correct_result = -1; 
			double max_val = -Double.MAX_VALUE;
			double[] output = neuralNet.getOutput();
			for (int i = 0; i < N_DIGIT; ++i){
				if (testSetRow.getDesiredOutput()[i] == 1){
					correct_result = i;
				}
				if (output[i] > max_val){
					max_val = output[i];
					result = i;
				}
			}
			//System.out.println("desired result: " + correct_result + ", result: " + result);
			if (result == correct_result){
				++score;
			}

		}
		System.out.println("score == " + score + "/" + (N_DIGIT * TEST_SIZE));
	}

	@Override	
	public void handleLearningEvent(LearningEvent event) {
		BackPropagation bp = (BackPropagation)event.getSource();
		System.out.println("iteration " + bp.getCurrentIteration() + "\t\ttotal network error "+ bp.getTotalNetworkError());
	} 

	public static void main(String[] args){
		new mnist_nn().run(args[0]);
	}

}
