package nl.peterbloem.vms;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.lilian.corpora.AdiosCorpus;
import org.lilian.corpora.Corpus;
import org.lilian.corpora.SequenceCorpus;
import org.lilian.corpora.SequenceIterator;
import org.lilian.corpora.WesternCorpus;
import org.nodes.data.Examples;

import nl.peterbloem.kit.FileIO;
import nl.peterbloem.kit.Series;
import nl.peterbloem.vms.LevelModel.RelevanceComparator;

public class LevelStatisticsExperiment
{
	public static final int MIN_FREQ = 5;
	private static final int MAX_TOKENS = 15;
	
	public static void main(String[] args)
		
	{
		
		ClassLoader classLoader = Examples.class.getClassLoader();
		File file = new File(classLoader.getResource("data/alice.txt").getFile());
				
		SequenceCorpus<String> vms = new WesternCorpus(file, false, true);
		
		SequenceIterator<String> it = vms.iterator();

		LevelModel<String> model = new LevelModel<String>(vms);
		
		List<String> tokens = model.tokens(MIN_FREQ);
		Collections.sort(tokens, 
				Collections.reverseOrder(model.new RelevanceComparator()));
		
		List<String> cTokens = new ArrayList<>(tokens.subList(0, MAX_TOKENS));
		
		Collections.sort(tokens, 
				Collections.reverseOrder(model.new SigNormComparator()));
		
		List<String> sTokens = new ArrayList<>(tokens.subList(0, MAX_TOKENS));
		
		for(int i : Series.series(MAX_TOKENS))
			System.out.println("<tr><td>"+cTokens.get(i)+"</td><td>"+sTokens.get(i)+"</td></tr>");
	
	}
}
