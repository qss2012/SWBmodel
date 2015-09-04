package models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.IntStream;

import utility.FuncUtils;

/**
 * SWB: A Java package for the SWB topic model
 * 
 * Implementation of the Special Word with Background (SWB) topic model, using
 * collapsed Gibbs sampling, as described in:
 * 
 * Chemudugunta, C., & Steyvers, P. S. M. (2007). Modelling General and Specific Aspects of 
 * Documents with a Probabilistic Topic Model. In Advances in Neural Information Processing
 * Systems 19: Proceedings of the 2006 Conference (Vol. 19, p. 241). MIT Press.
 * 
 * @author: Sultan Alqahtani
 */

public class GibbsSamplingSWB
{
	public double alpha; // Hyper-parameter alpha
	public double[] betas; // Hyper-parameter betas
		// Sultan added
		public double gamma; // Hyper-parameter gamma
	public int numTopics; // Number of topics
	public int numIterations; // Number of Gibbs sampling iterations
	public int topWords; // Number of most probable words for each topic

	public double alphaSum; // alpha * numTopics
	public double[] betaSum; // beta[i] * vocabularySize
		// Sultan added
		public double gammaSum; // gamma * 3 -> three means words categories (topic, special-words, or background word)

	public List<List<Integer>> corpus; // Word ID-based corpus
	public List<List<Integer>> topicAssignments; // Topics assignments for words in the corpus
		//Sultan added
		public List<List<Integer>> wordTypeAssignments; // x assignments for words in the corpus
	
	public int numDocuments; // Number of documents in the corpus
	public int numWordsInCorpus; // Number of words in the corpus

	public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
														// given a word
	public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
														// given an ID
	public int vocabularySize; // The number of word types in the corpus

	// numDocuments * numTopics matrix
	// Given a document: number of its words assigned to each topic
	public int[][] docTopicCount;
	// Number of words in every document
	public int[] sumDocTopicCount;
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type assigned to the topic
	public int[][] topicWordCount;
	// Total number of words assigned to a topic
	public int[] sumTopicWordCount;
		// Sultan added
		// vocabularySize * numDocuments matrix
		// Given a document: number of times word w is assigned to o the special-words distribution of document d 
		public int[][] docWordCount;
		// Total number of special-words  in every document
		public int[] sumDocWordCount;
		// vocabularySize 
		// Given a document: number of times word w is assigned to the background distribution
		public int [][] wordCount;
		public int [] sumWordCount;
		
		// Sultan added
		// Number of words in document d 
		public int Nd[];
		// Number of words in document d assigned to the latent topics -> similar to sumDocTopicCount
		public int Nd0[];
		// Number of words in document d assigned to special words
		public int Nd1[];
		// Number of words in document d assigned to  background component
		// Nd2 = Nd - ( Nd0 + Nd1)
		public int Nd2[];  
		
		// Sultan added
		public int wordType[];
		
		// Sultan added
		// Double array used to sample x -> x is sampled from a document-specific multinomial LEMDA, which in turn has a symmetric Dirichlet prior, GAMMA
		public double x[];

	// Double array used to sample a topic
	public double[] multiPros;

	// Path to the directory containing the corpus
	public String folderPath;
	// Path to the topic modeling corpus
	public String corpusPath;

	public String expName = "LDAmodel";
	public String orgExpName = "LDAmodel";
	public String tAssignsFilePath = "";
	public int savestep = 0;


	public GibbsSamplingSWB(String pathToCorpus, int inNumTopics,
		double inAlpha, double[] inBetas,double gamma, int inNumIterations, int inTopWords,
		String inExpName)
		throws Exception
	{
		this(pathToCorpus, inNumTopics, inAlpha, inBetas,gamma, inNumIterations,
			inTopWords, inExpName, "");
	}

	public GibbsSamplingSWB(String pathToCorpus, int inNumTopics,
		double inAlpha, double[] inBeta,double gamma, int inNumIterations, int inTopWords,
		String inExpName, String pathToTAfile)
		throws Exception
	{

		alpha = inAlpha;
		this.betas = inBeta;
			// Sultan added
			this.gamma = gamma;
		numTopics = inNumTopics;
		numIterations = inNumIterations;
		topWords = inTopWords;
		expName = inExpName;
		orgExpName = expName;
		corpusPath = pathToCorpus;
		folderPath = pathToCorpus.substring(
			0,
			Math.max(pathToCorpus.lastIndexOf("/"),
				pathToCorpus.lastIndexOf("\\")) + 1);

		System.out.println("Reading topic modeling (SWB) corpus: " + pathToCorpus);

		word2IdVocabulary = new HashMap<String, Integer>();
		id2WordVocabulary = new HashMap<Integer, String>();
		corpus = new ArrayList<List<Integer>>();
		numDocuments = 0;
		numWordsInCorpus = 0;

		BufferedReader br = null;
		try {
			int indexWord = -1;
			br = new BufferedReader(new FileReader(pathToCorpus));
			for (String doc; (doc = br.readLine()) != null;) {

				if (doc.trim().length() == 0)
					continue;

				String[] words = doc.trim().split("\\s+");
				List<Integer> document = new ArrayList<Integer>();

				for (String word : words) {
					if (word2IdVocabulary.containsKey(word)) {
						document.add(word2IdVocabulary.get(word));
					}
					else {
						indexWord += 1;
						word2IdVocabulary.put(word, indexWord);
						id2WordVocabulary.put(indexWord, word);
						document.add(indexWord);
					}
				}

				numDocuments++;
				numWordsInCorpus += document.size();
				corpus.add(document);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}

		vocabularySize = word2IdVocabulary.size(); // vocabularySize = indexWord
		docTopicCount = new int[numDocuments][numTopics];
		topicWordCount = new int[numTopics][vocabularySize];
		sumDocTopicCount = new int[numDocuments];
		sumTopicWordCount = new int[numTopics];
		
			// Sultan added
			docWordCount = new int[numDocuments][vocabularySize];
			sumDocWordCount = new int[numDocuments];
			wordCount = new int[numDocuments][vocabularySize];
			sumWordCount = new int[vocabularySize];
			
			//Sultan added
			Nd = new int[numDocuments];
			Nd0 = new int[vocabularySize];
			Nd1 = new int[vocabularySize];
			Nd2 = new int[vocabularySize];
			
			//Sultan 
			wordType = new int[vocabularySize];
			
			//Sultan added
			x = new double[3];
			for (int j = 0; j <= 2; j++) {
				x[j] = 1.0/3;
			}

		multiPros = new double[numTopics];
		for (int i = 0; i < numTopics; i++) {
			multiPros[i] = 1.0 / numTopics;
		}

		alphaSum = numTopics * alpha;
		this.betaSum = new double[3];
		for(int i = 0; i<3 ; i++)
			betaSum[i] = betas[i] * vocabularySize;
			// Sultan added
			gammaSum = gamma * 3;

		System.out.println("Corpus size: " + numDocuments + " docs, "
			+ numWordsInCorpus + " words");
		System.out.println("Vocabuary size: " + vocabularySize);
		System.out.println("Number of topics: " + numTopics);
		System.out.println("alpha: " + alpha);
		for(int i=0 ; i<3 ; i++) System.out.println("beta: " + betas[i]);
		System.out.println("Number of sampling iterations: " + numIterations);
		System.out.println("Number of top topical words: " + topWords);

		tAssignsFilePath = pathToTAfile;
		if (tAssignsFilePath.length() > 0)
			initialize(tAssignsFilePath);
		else
			initialize();
	}

	/**
	 * Randomly initialize topic assignments
	 */
	public void initialize() throws IOException {
		System.out.println("Randomly initializing topic assignments ...");

		topicAssignments = new ArrayList<List<Integer>>();
		// Sultan added
		wordTypeAssignments = new ArrayList<List<Integer>>();

		for (int i = 0; i < numDocuments; i++) {
			List<Integer> topics = new ArrayList<Integer>();
			// Sultan added
			List<Integer> xValues = new ArrayList<Integer>();
			int docSize = corpus.get(i).size();
			for (int j = 0; j < docSize; j++) {
				int topic = FuncUtils.nextDiscrete(multiPros); // Sample a topic
				// Sultan added
				int xValue = FuncUtils.nextDiscrete(x); // Sample x (x will have
														// values 0, 1, or 2);
				if (xValue == 0) {
					Nd0[corpus.get(i).get(j)] += 1;
					// Increase counts
					docTopicCount[i][topic] += 1;
					topicWordCount[topic][corpus.get(i).get(j)] += 1;
					sumDocTopicCount[i] += 1;
					sumTopicWordCount[topic] += 1;
				} else if (xValue == 1) {
					Nd1[corpus.get(i).get(j)] += 1;
					// Increase counts
					docWordCount[i][corpus.get(i).get(j)] += 1;
					sumDocWordCount[i] += 1;
				} else if (xValue == 2) {
					Nd2[corpus.get(i).get(j)] += 1;
					// Increase counts
					wordCount[i][corpus.get(i).get(j)] += 1;
					sumWordCount[corpus.get(i).get(j)] += 1;
				}
				xValues.add(xValue);
				topics.add(topic);

				Nd[i] += Nd0[corpus.get(i).get(j)] + Nd1[corpus.get(i).get(j)]
						+ Nd2[corpus.get(i).get(j)];
			}

			topicAssignments.add(topics);
			// Sultan added
			wordTypeAssignments.add(xValues);
		}

	}

	/**
	 * Initialize topic assignments from a given file
	 */
	public void initialize(String pathToTopicAssignmentFile)
	{
		System.out.println("Reading topic-assignment file: "
			+ pathToTopicAssignmentFile);

		topicAssignments = new ArrayList<List<Integer>>();

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(pathToTopicAssignmentFile));
			int docID = 0;
			int numWords = 0;
			for (String line; (line = br.readLine()) != null;) {
				String[] strTopics = line.trim().split("\\s+");
				List<Integer> topics = new ArrayList<Integer>();
				for (int j = 0; j < strTopics.length; j++) {
					int topic = new Integer(strTopics[j]);
					// Increase counts
					docTopicCount[docID][topic] += 1;
					topicWordCount[topic][corpus.get(docID).get(j)] += 1;
					sumDocTopicCount[docID] += 1;
					sumTopicWordCount[topic] += 1;

					topics.add(topic);
					numWords++;
				}
				topicAssignments.add(topics);
				docID++;
			}

			if ((docID != numDocuments) || (numWords != numWordsInCorpus)) {
				System.out
					.println("The topic modeling corpus and topic assignment file are not consistent!!!");
				throw new Exception();
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void inference()
		throws IOException
	{
		System.out.println("Running Gibbs sampling inference: ");

		for (int iter = 1; iter <= numIterations; iter++) {

			System.out.println("\tSampling iteration: " + (iter));
		//	System.out.println("\t\tPerplexity: " + computePerplexity());

			sampleInSingleIteration();

			if ((savestep > 0) && (iter % savestep == 0)
				&& (iter < numIterations)) {
				System.out.println("\t\tSaving the output from the " + iter
					+ "^{th} sample");
				expName = orgExpName + "-" + iter;
				write();
			}
		}
		expName = orgExpName;

		writeParameters();
		System.out.println("Writing output from the last sample ...");
		write();

		System.out.println("Sampling completed!");

	}

	public void sampleInSingleIteration() {
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				// Get current word and its topic
				int topic = topicAssignments.get(dIndex).get(wIndex);
				int word = corpus.get(dIndex).get(wIndex);
				// Sultan added
				int xValue = wordTypeAssignments.get(dIndex).get(wIndex);
				Nd[dIndex] -= 1;
				if (xValue == 0) {
					Nd0[word] -= 1;
					docTopicCount[dIndex][topic] -= 1;
					topicWordCount[topic][word] -= 1;
					sumDocTopicCount[dIndex] -= 1;
					sumTopicWordCount[topic] -= 1;
					// Sample x and topic
					for (int wIndex_ = 0; wIndex_ <= 2; wIndex_++) {
						x[wIndex_] = (Nd0[word] + gamma) / (Nd[dIndex] + 3 * gamma);
						for (int tIndex = 0; tIndex < numTopics; tIndex++) {
							multiPros[tIndex] = ((Nd0[word] + gamma) / (Nd[dIndex] + 3 * gamma))
									* ((docTopicCount[dIndex][tIndex] + alpha) / (sumDocTopicCount[dIndex] + alphaSum))
									* ((topicWordCount[tIndex][word] + betas[0]) / (sumTopicWordCount[tIndex] + betaSum[0]));
						}
					}
					xValue = FuncUtils.nextDiscrete(x);
					topic = FuncUtils.nextDiscrete(multiPros);
					// Increase counts
					Nd0[word] += 1;
					docTopicCount[dIndex][topic] += 1;
					topicWordCount[topic][word] += 1;
					sumDocTopicCount[dIndex] += 1;
					sumTopicWordCount[topic] += 1;
					// Update x assignments
					wordTypeAssignments.get(dIndex).set(wIndex, xValue);
					// Update topic assignments
					topicAssignments.get(dIndex).set(wIndex, topic);
				}else if (xValue == 1) {
					// Decrease counts
					Nd1[word] -= 1;
					docWordCount[dIndex][word] -= 1;
					sumDocWordCount[dIndex] -= 1;
					// Sample x
					for (int wIndex_ = 0; wIndex_ <= 2; wIndex_++) {
						x[wIndex_] = ((Nd1[word] + gamma) / (Nd[dIndex] + 3 * gamma))
								* ((docWordCount[dIndex][word] + betas[1]) / (sumDocWordCount[dIndex] + betaSum[1]));
					}
					xValue = FuncUtils.nextDiscrete(x);
					// Increase counts
					Nd1[word] += 1;
					docWordCount[dIndex][word] += 1;
					sumDocWordCount[dIndex] += 1;
					// Update x assignments
					wordTypeAssignments.get(dIndex).set(wIndex, xValue);
				} else  if (xValue == 2) {
					// Decrease counts
					Nd2[word] -= 1;
					wordCount[dIndex][word] -= 1;
					sumWordCount[word] -= 1;
					// Sample x
					for (int wIndex_ = 0; wIndex_ <= 2; wIndex_++) {
						x[wIndex_] = ((Nd2[word] + gamma) / (Nd[dIndex] + 3 * gamma))
								* ((wordCount[dIndex][word] + betas[2]) / (sumDocWordCount[dIndex] + betaSum[2]));
					}
					xValue = FuncUtils.nextDiscrete(x);
					// Increase counts
					Nd2[word] += 1;
					wordCount[dIndex][word] += 1;
					sumWordCount[word] += 1;
					// Update x assignments
					wordTypeAssignments.get(dIndex).set(wIndex, xValue);
				}
			}
		}
	}

//	public double computePerplexity() {
//		double logliCorpus = 0.0;
//		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
//			int docSize = corpus.get(dIndex).size();
//			double logliDoc = 0.0;
//			for (int wIndex = 0; wIndex < docSize; wIndex++) {
//				int word = corpus.get(dIndex).get(wIndex);
//				double likeWord = 0.0;
//				for (int tIndex = 0; tIndex < numTopics; tIndex++) {
//					likeWord += ((docTopicCount[dIndex][tIndex] + alpha) / (sumDocTopicCount[dIndex] + alphaSum))
//							* ((topicWordCount[tIndex][word] + betas[0]) / (sumTopicWordCount[tIndex] + betaSum[0]));
//				}
//				logliDoc += Math.log(likeWord);
//			}
//			logliCorpus += logliDoc;
//		}
//		double perplexity = Math.exp(-1.0 * logliCorpus / numWordsInCorpus);
//		if (perplexity < 0)
//			throw new RuntimeException("Illegal perplexity value: "
//					+ perplexity);
//		return perplexity;
//	}

	public void writeParameters()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".paras"));
		writer.write("-model" + "\t" + "LDA");
		writer.write("\n-corpus" + "\t" + corpusPath);
		writer.write("\n-ntopics" + "\t" + numTopics);
		writer.write("\n-alpha" + "\t" + alpha);
		writer.write("\n-beta" + "\t" + "[" + betas[0] + "," + betas[1] + "]");
		writer.write("\n-niters" + "\t" + numIterations);
		writer.write("\n-twords" + "\t" + topWords);
		writer.write("\n-name" + "\t" + expName);
		if (tAssignsFilePath.length() > 0)
			writer.write("\n-initFile" + "\t" + tAssignsFilePath);
		if (savestep > 0)
			writer.write("\n-sstep" + "\t" + savestep);

		writer.close();
	}

	public void writeDictionary()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".vocabulary"));
		for (String word : word2IdVocabulary.keySet()) {
			writer.write(word + " " + word2IdVocabulary.get(word) + "\n");
		}
		writer.close();
	}

	public void writeIDbasedCorpus()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".IDcorpus"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(corpus.get(dIndex).get(wIndex) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicAssignments()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".topicAssignments"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(topicAssignments.get(dIndex).get(wIndex) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopTopicalWords()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".topWords"));

		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			writer.write("Topic" + new Integer(tIndex) + ":");

			Map<Integer, Integer> wordCount = new TreeMap<Integer, Integer>();
			for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {
				wordCount.put(wIndex, topicWordCount[tIndex][wIndex]);
			}
			wordCount = FuncUtils.sortByValueDescending(wordCount);

			Set<Integer> mostLikelyWords = wordCount.keySet();
			int count = 0;
			for (Integer index : mostLikelyWords) {
				if (count < topWords) {
					writer.write(" " + id2WordVocabulary.get(index));
					count += 1;
				}
				else {
					writer.write("\n\n");
					break;
				}
			}
		}
		writer.close();
	}

	public void writeTopicWordPros()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".phi"));
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < vocabularySize; j++) {
				double pro = (topicWordCount[i][j] + betas[0])
					/ (sumTopicWordCount[i] + betaSum[0]);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicWordCount()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".WTcount"));
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < vocabularySize; j++) {
				writer.write(topicWordCount[i][j] + " ");
			}
			writer.write("\n");
		}
		writer.close();

	}

	public void writeDocTopicPros()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".theta"));
		for (int i = 0; i < numDocuments; i++) {
			for (int j = 0; j < numTopics; j++) {
				double pro = (docTopicCount[i][j] + alpha)
					/ (sumDocTopicCount[i] + alphaSum);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeDocTopicCount()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".DTcount"));
		for (int i = 0; i < numDocuments; i++) {
			for (int j = 0; j < numTopics; j++) {
				writer.write(docTopicCount[i][j] + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void write()
		throws IOException
	{
		writeTopTopicalWords();
		writeDocTopicPros();
		writeTopicAssignments();
		writeTopicWordPros();
	}

	public static void main(String args[])
		throws Exception
	{
		String pathToCorpus = "data/corpus.txt";
		double[] betas = {0.01, 0.01,0.0001};
		double alpha = 0.1;
		double gamma = 0.3;
		int iteration = 1000;
		int topWords = 100;
		int numTopics = 25;
		
		GibbsSamplingSWB swb = new GibbsSamplingSWB(pathToCorpus, numTopics, alpha,
				betas,gamma, iteration, topWords, "testSWB");
		swb.inference();

	}
}
