import java.io.File;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.Evaluation;

public class Test {
	/* note: file 750 appears to be truncated */
	private static final int NUM_TRAIN = 11, TRAIN_SIZE[] = {50, 100, 200, 300, 400, 500, 1000, 1250, 1500, 1750, 2000};
	private static final String TRAIN_PREFIX = "train" + File.separator + "spam_train_", TRAIN_SUFFIX = ".arff", TEST = "test" + File.separator + "spam_test.arff"; 
	public static void main(String args[]) throws Exception {
		final ArffLoader eval_ld = new ArffLoader();
		eval_ld.setFile(new File(TEST));
		final Instances testInstances = eval_ld.getDataSet();
		testInstances.setClassIndex(testInstances.numAttributes() - 1);
		for (int n = 0; n < NUM_TRAIN; ++n) {
			System.out.println("n == " + n + ", TRAIN_SIZE[n] == " + TRAIN_SIZE[n]);
			final ArffLoader ld = new ArffLoader();
			ld.setFile(new File(TRAIN_PREFIX + TRAIN_SIZE[n] + TRAIN_SUFFIX));
			final Instances trainInstances = ld.getDataSet();
			final NaiveBayesMultinomial nb = new NaiveBayesMultinomial();
			Instance inst;
			trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
			nb.buildClassifier(trainInstances);
			/*
			while ((inst = ld.getNextInstance(trainInstances)) != null) nb.updateClassifier(inst);
			*/
			final Evaluation eval = new Evaluation(trainInstances);
			/* System.out.println(nb); */
			eval.evaluateModel(nb, testInstances);
			System.out.println(eval.pctCorrect());
		}
	}
}
