package nl.peterbloem.vms;

import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.reverseOrder;
import static nl.peterbloem.kit.Series.series;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.lilian.corpora.SequenceCorpus;
import org.lilian.corpora.SequenceIterator;
import org.lilian.corpora.WesternCorpus;
import org.lilian.corpora.wrappers.Characters;
import org.nodes.data.Examples;

import nl.peterbloem.kit.Global;
import nl.peterbloem.kit.MaxObserver;
import nl.peterbloem.kit.Series;
import nl.peterbloem.vms.LevelModel.RelevanceComparator;
import nl.peterbloem.vms.LevelModel.SigNormComparator;

public class WhiteSpaceExperiment
{
	public static final int MIN = 2;
	public static final int MAX = 35;
	public static final int MIN_FREQ = 5;
	public static final int MAX_CHILDREN = 1;
	private static final int MAX_TOKENS = 15;
	private static final int MAX_STARTS = 10000;
	private static final double DIFF = 2.5;
	
	public static void main(String[] args) 
		throws IOException
	{
		ClassLoader classLoader = Examples.class.getClassLoader();
		File file = new File(classLoader.getResource("data/eva.takahashi.txt").getFile());
		
		SequenceCorpus<String> vms = new WesternCorpus(file, false, true);
		
		Set<String> wordSet = new LinkedHashSet<String>();
		for(String word : vms)
			if(word.length() > 2)
				wordSet.add(word);
		List<String> words = new ArrayList<>(wordSet);
		Collections.sort(words, new Comparator<String>()
		{
			@Override
			public int compare(String o1, String o2)
			{
				return - Integer.compare(o1.length(), o2.length());
			}
		});
						
		vms = Characters.wrap(vms);
		
		Global.log().info("Collecting statistics");
		
		SequenceModel<String> seq = new SequenceModel<>(vms, MIN, MAX);
		
//		System.out.println(seq.frequency(token("energy")));
//		System.out.println(seq.sigNorm(token("energy")));
//		System.out.println(seq.relevance(token("energy")));
		
		Global.log().info("Searching for tokens");
		
		MaxObserver<List<String>> observer = 
				new MaxObserver<>(200, seq.new RelevanceComparator());
		
		for(int l : Series.series(5, MAX))
			observer.observe(seq.model(l).tokens());
			
		List<List<String>> tokens = new ArrayList<>();
		
		for(List<String> token : observer.elements())
		{
			boolean skip = false;
			List<String> rem = null;
			
			for(List<String> better : tokens)
			{				
				if(toString(better).contains(toString(token)))
				{
					skip = true;
					break;
				}
			
				if(toString(token).contains(toString(better)))
					if(seq.relevance(better) - seq.relevance(token) < DIFF)
						rem = better;
			}
			
			if(! skip)
			{
				tokens.remove(rem);
				tokens.add(token);
			}
		}
		
//		double c0 = 1.1645; // 0.05
//		double c0 = 1.96;   // 0.025
//		double c0 = 2.33;   // 0.01
//		double c0 = 2.58;   // 0.005
//		double c0 = 3.09;   // 0.001
//		double c0 = 3.719;  // 0.0001
		
//		for(int l0 : Series.series(MIN, MAX))
//		{
//			List<List<String>> tok = seq.model(l0).tokens();
//			
//			Collections.sort(tok, 
//				Collections.reverseOrder(seq.new RelevanceComparator()));
//
//			int max = 0;
//			while(seq.relevance(tok.get(max ++)) > c0);
//			
//			tok = tok.subList(0, max);
//			
//			System.out.println(l0 + " " + tok.size());
//									
//			for(List<String> start : tok)
//			{
//				boolean print = false;
//				if(toString(start).equals("energy"))
//					print = true;
//				
//				List<List<String>> ancestors = new ArrayList<>(MAX);
//				ancestors(start, ancestors, seq);
//				
//				if(print)
//					System.out.println(ancestors);
//								
//				List<String> longest = Collections.emptyList();
//				for(List<String> anc : ancestors)
//					if(anc.size() > longest.size() && seq.relevance(anc) >= c0)
//						longest = anc;
//				
//				if(print)
//					System.out.println(longest + " " + seq.relevance(longest));
//				
//				if(! longest.isEmpty())
//					tokenSet.add(longest);
//			}
//		}

		for(List<String> token : tokens)
		{
			String highlighted = highlight(toString(token), words);
			
			System.out.println(String.format(
					"<tr><td> %s \t</td><td> %d </td><td>%.2f </td><td> %.2f </td></tr>", 
					highlighted, (int)seq.frequency(token), seq.sigNorm(token), seq.relevance(token)));
		}
	}

	private static String highlight(String string, List<String> words)
	{
		
		String rest = string;
		String prefix = "";
		String suffix = "";
		
		for(String word : words)
			if(rest.endsWith(word))
			{
				suffix = word;
				rest = rest.substring(0, rest.length() - suffix.length());
				
				break;
			}
		
		for(String word : words)
			if(rest.startsWith(word))
			{
				prefix = word;
				rest = rest.substring(prefix.length());
				
				break;
			}
		
		String result = "<span>" + rest + "</span>";
		
		if(prefix.length() > 0)
			result = "<strong>" + prefix + "</strong>" + result;
		
		if(suffix.length() > 0)
			result += "<strong>" + suffix + "</strong>";
		
		return  result;
	}

	/**
	 * find all leaf ancestors of this token
	 * @param token
	 * @param seq
	 * @return
	 */
	private static Collection<List<String>> findOne(List<String> token, SequenceModel<String> seq)
	{
		if(token.size() == MAX)
			return asList(token);
		
		
		List<List<String>> result = new ArrayList<>();
		
		double relevance = seq.relevance(token);
		if(Double.isNaN(relevance))
			return Collections.emptyList();
						
		for(List<String> child : seq.children(token))
		{
			double cRelevance = seq.relevance(child);
			if((! Double.isNaN(cRelevance)) && cRelevance > relevance)
				result.add(child);
		}
		
		if(result.isEmpty())
			return asList(token);
		
		if(MAX_CHILDREN != -1 && result.size() > MAX_CHILDREN)
			result = MaxObserver.quickSelect(MAX_CHILDREN, result,  
					Collections.reverseOrder(seq.new RelevanceComparator()), false);
		
		return find(result, seq);
	}
	
	
	private static Collection<List<String>> find(List<List<String>> tokens, SequenceModel<String> seq)
	{
		Set<List<String>> result = new LinkedHashSet<>();
		
		for(List<String> token : tokens)
			result.addAll(findOne(token, seq));
		
		return result;
	}
	
	private static void ancestors(List<String> start, List<List<String>> result, SequenceModel<String> seq) 
	{
		result.add(start);
		
		if(start.size() == MAX)
			return;
				
		List<String> best = null;
		double bestRelevance = Double.NEGATIVE_INFINITY;
		
		for(List<String> child : seq.children(start))
		{
			double relevance = seq.relevance(child);
			if((! Double.isNaN(relevance)) && relevance >= bestRelevance)
			{
				best = child;
				bestRelevance = relevance;
			}
		}
		
		ancestors(best, result, seq);
	}
	
	private static String toString(List<String> token)
	{
		StringBuilder builder = new StringBuilder();
		
		for(String t : token)
			builder.append(t);
		
		return builder.toString();
	}
	
	
	private static List<String> token(String string)
	{
		List<String> result = new ArrayList<>(string.length());
		
		for(int i : series(string.length()))
			result.add(string.charAt(i)+"");
		return result;
	}
	
	
}
