import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.conversion.meanshift.InputDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.clustering.ClusterDumper;

import cern.colt.Arrays;

public class BuildCodebook extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(BuildCodebook.class);

	// Mapper: emits (token, 1) for every word occurrence.
	private static class MyMapper extends
			Mapper<LongWritable, Text, Text, IntWritable> {

		// Reuse objects to save overhead of object creation.
		private final static IntWritable ONE = new IntWritable(1);
		private final static Text WORD = new Text();

		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String line = ((Text) value).toString();
			StringTokenizer itr = new StringTokenizer(line);
			while (itr.hasMoreTokens()) {
				WORD.set(itr.nextToken());
				context.write(WORD, ONE);
			}
		}
	}

	// Reducer: sums up all the counts.
	private static class MyReducer extends
			Reducer<Text, IntWritable, Text, IntWritable> {

		// Reuse objects.
		private final static IntWritable SUM = new IntWritable();

		@Override
		public void reduce(Text key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {
			// Sum up values.
			Iterator<IntWritable> iter = values.iterator();
			int sum = 0;
			while (iter.hasNext()) {
				sum += iter.next().get();
			}
			SUM.set(sum);
			context.write(key, SUM);
		}
	}

	public void TransformVectorsToSequence(Configuration conf,
			String inputPath, String outputPath) throws IOException {
		FileSystem fs = FileSystem.get(conf);
		Path inPath = new Path(inputPath);
		Path outPath = new Path(outputPath);

		SequenceFile.Writer writer = SequenceFile.createWriter(conf,
				SequenceFile.Writer.file(outPath),
				SequenceFile.Writer.keyClass(Text.class),
				SequenceFile.Writer.valueClass(VectorWritable.class));

		Text key = new Text();
		VectorWritable value = new VectorWritable();

		FSDataInputStream fdstream = fs.open(inPath);
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				fdstream));

		String line = "";
		int id = 0;
		while ((line = reader.readLine()) != null) {
			StringTokenizer token = new StringTokenizer(line);
			int numTokens = token.countTokens();
			NamedVector vec = new NamedVector();
			for (int i = 0; i < numTokens; ++i)
				vec.setQuick(i, Double.parseDouble(token.nextToken()));
			key.set(String.valueOf(++id));
			value.set(vec);
			writer.append(key, value);
		}
		writer.close();
	}

	public void KMeansByMahout(Configuration conf, String inputPath,
			String outputPath) throws Exception {
		
		DistanceMeasure measure = new EuclideanDistanceMeasure();
		int K = 1024;

		// Read from text input data and transform it to Sequence File
		//TransformVectorsToSequence(conf, inputPath, sequencePath);
		Path input = new Path(inputPath);
		String sequencePath = "codebook/input-serial";
		Path sequence = new Path(sequencePath);
		FileSystem.get(conf).delete(sequence, true);
		
	    LOG.info("Preparing Input");
		InputDriver.runJob(input, sequence);

		Path output = new Path(outputPath);
		FileSystem.get(conf).delete(output, true);
		
		// Initial clustering
		LOG.info("Running random seed to get initial clusters");
		Path clusters = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);
	    clusters = RandomSeedGenerator.buildRandom(conf, sequence, clusters, K, measure);
	    
		// Kmeans clustering

		double convergenceDelta = 1e-3;
		int maxIterations = 100;
	    LOG.info("Running KMeans");
		KMeansDriver.run(conf, sequence, clusters, output,
				measure, convergenceDelta, maxIterations, true, 0.0, false);

	    // run ClusterDumper
	    ClusterDumper clusterDumper = new ClusterDumper(new Path(output, "clusters-*-final"), new Path(output,
	        "clusteredPoints"));
	    clusterDumper.printClusters(null);
	}

	/**
	 * Creates an instance of this tool.
	 */
	public BuildCodebook() {
	}

	private static final String INPUT = "input";
	private static final String OUTPUT = "output";

	/**
	 * Runs this tool.
	 */
	@SuppressWarnings({ "static-access" })
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("input path").create(INPUT));
		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("output path").create(OUTPUT));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: "
					+ exp.getMessage());
			return -1;
		}

		if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}

		String inputPath = cmdline.getOptionValue(INPUT);
		String outputPath = cmdline.getOptionValue(OUTPUT);

		LOG.info("Tool: " + BuildCodebook.class.getSimpleName());

		Configuration conf = getConf();

		KMeansByMahout(conf, inputPath, outputPath);

		long startTime = System.currentTimeMillis();
		LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new BuildCodebook(), args);
	}
}
