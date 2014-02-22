import java.util.Scanner;
import weka.core.FastVector;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
/* import weka.classifiers.functions.IBk; */
import weka.classifiers.lazy.IBk;
import weka.classifiers.Evaluation;

public class Test {

	private static final int NUM_VAL = 5, OPEN_PRICE_IDX = 0, MAX_PRICE_IDX = 1, MIN_PRICE_IDX = 2, CLOSE_PRICE_IDX = 3, VOL_IDX = 4, HIST_SIZE = 60, NUM_TRAIN = 3000, NUM_TEST = 500;
	private static final double EPS = 1e-9;
	private static final String OUTCOME = "OUTCOME", LOSS = "LOSS", PROFIT = "PROFIT";
	private static final Scanner sc = new Scanner(System.in);
	private static final FastVector attInfo = new FastVector(NUM_VAL + 3);

	static {
		final FastVector outcome = new FastVector();
		outcome.addElement(LOSS);
		outcome.addElement(PROFIT);
		for (int k = 0; k < (HIST_SIZE - 1) * (NUM_VAL + 2); ++k) attInfo.addElement(new Attribute(Integer.toString(k))); 
		attInfo.addElement(new Attribute(OUTCOME, outcome));
	}

	private double val[][] = new double[NUM_VAL][HIST_SIZE], n_val[][] = new double[NUM_VAL + 2][HIST_SIZE]; 
	private Instances trainInstances = new Instances("trainInstances", attInfo, NUM_TRAIN), testInstances = new Instances("testInstances", attInfo, NUM_TEST);
	private IBk model = new IBk(); 

	private void normalize_min_max(final int k) {
		for (int f = 0; f < NUM_VAL; ++f) {
			double min = Double.MAX_VALUE, max = 0.0, diff;
			for (int K = 0; K + 1 < HIST_SIZE; ++K) {
				if (n_val[f][(k + K) % HIST_SIZE] < min) min = n_val[f][(k + K) % HIST_SIZE]; 
				if (n_val[f][(k + K) % HIST_SIZE] > max) max = n_val[f][(k + K) % HIST_SIZE]; 
			}
			if ((diff = max - min) > EPS) for (int K = 0; K + 1 < HIST_SIZE; ++K) n_val[f][(k + K) % HIST_SIZE] = (n_val[f][(k + K) % HIST_SIZE] - min) / (max - min);
			else for (int K = 0; K + 1 < HIST_SIZE; ++K) n_val[f][(k + K) % HIST_SIZE] = 0.0;
		}
	}

	private void read_val(final int k) {
		double diff;
		final Scanner ln = new Scanner(sc.nextLine());
		ln.useDelimiter(",");
		for (int n = 0; n < NUM_VAL; ++n) n_val[n][k] = val[n][k] = ln.nextDouble();
		if ((diff = val[MAX_PRICE_IDX][k] - val[MIN_PRICE_IDX][k]) > EPS) {
			n_val[NUM_VAL][k] = (val[OPEN_PRICE_IDX][k] - val[MIN_PRICE_IDX][k]) / diff;
			n_val[NUM_VAL + 1][k] = (val[CLOSE_PRICE_IDX][k] - val[MIN_PRICE_IDX][k]) / diff;
		}else {
			n_val[NUM_VAL + 1][k] = n_val[NUM_VAL][k] = 0.0;
		}
	}

	private void train_model() throws Exception {
		double ct[] = new double[(HIST_SIZE - 1) * (NUM_VAL + 2) + 1];
		trainInstances.setClassIndex((HIST_SIZE - 1) * (NUM_VAL + 2));
		for (int k = 0; k < HIST_SIZE; ++k) read_val(k);
		for (int k = 0; k < NUM_TRAIN; ++k) {
			normalize_min_max(k);
			for (int idx = 0, K = 0; K + 1 < HIST_SIZE; ++K) for (int f = 0; f < NUM_VAL + 2; ++f, ++idx) ct[idx] = n_val[f][(k + K) % HIST_SIZE]; 
			double diff = val[CLOSE_PRICE_IDX][(k + HIST_SIZE - 1) % HIST_SIZE] - val[OPEN_PRICE_IDX][(k + HIST_SIZE - 1) % HIST_SIZE];
			if (diff < 0) diff = -diff;
			Instance current = new Instance(diff * val[VOL_IDX][(k + HIST_SIZE - 1) % HIST_SIZE], ct);
			current.setDataset(trainInstances);
			current.setClassValue((val[CLOSE_PRICE_IDX][(k + HIST_SIZE - 1) % HIST_SIZE] < val[OPEN_PRICE_IDX][(k + HIST_SIZE - 1) % HIST_SIZE] ? LOSS : PROFIT));
			trainInstances.add(current);
			read_val(k % HIST_SIZE);
		}
		model.buildClassifier(trainInstances);	
	}

	private void test_model() throws Exception {
		double gain = 0.0;
		double ct[] = new double[(HIST_SIZE - 1) * (NUM_VAL + 2) + 1];
		testInstances.setClassIndex((HIST_SIZE - 1) * (NUM_VAL + 2));
		final Evaluation eval = new Evaluation(trainInstances);
		for (int k = 0; k < HIST_SIZE; ++k) read_val(k);
		for (int k = 0; k < NUM_TEST; ++k) {
			normalize_min_max(k);
			for (int idx = 0, K = 0; K + 1 < HIST_SIZE; ++K) for (int f = 0; f < NUM_VAL + 2; ++f, ++idx) ct[idx] = n_val[f][(k + K) % HIST_SIZE]; 
			double diff = val[CLOSE_PRICE_IDX][(k + HIST_SIZE - 1) % HIST_SIZE] - val[OPEN_PRICE_IDX][(k + HIST_SIZE - 1) % HIST_SIZE], abs_diff = diff < 0.0 ? -diff : diff;
			Instance current = new Instance(abs_diff * val[VOL_IDX][(k + HIST_SIZE - 1) % HIST_SIZE], ct);
			current.setDataset(testInstances);
			current.setClassValue(diff < 0.0 ? LOSS : PROFIT);
			testInstances.add(current);
			gain += eval.evaluateModelOnce(model, current) * diff * val[VOL_IDX][(k + HIST_SIZE - 1) % HIST_SIZE];
			read_val(k % HIST_SIZE);
		}
		eval.evaluateModel(model, testInstances);
		System.out.println("weighted accuracy: " + eval.pctCorrect() + "\nnet gain: " + gain);
	}

	public static void main(String args[]) throws Exception {
		Test t = new Test();
		t.train_model();
		t.test_model();
	}

}
