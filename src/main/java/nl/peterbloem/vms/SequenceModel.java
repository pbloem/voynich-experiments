package nl.peterbloem.vms;

import java.io.*;
import java.util.*;

import org.lilian.corpora.*;
import org.lilian.models.*;

import nl.peterbloem.kit.Global;
import nl.peterbloem.vms.LevelModel;

public class SequenceModel<T> 
{
	
	private List<List<T>> buffers;
	public List<LevelModel<List<T>>> models;
	private HashSet<T> alphabet= new HashSet<T>();
	
	private int min, max;
	
	public SequenceModel(int min, int max)
	{
		this.min = min;
		this.max = max;
		
		buffers = new ArrayList<List<T>>(max + 1);
		models  = new ArrayList<LevelModel<List<T>>>(max + 1);		
		
		for(int i = min; i < max + 1; i++)
		{
			buffers.add(new LinkedList<T>());
			models.add(new LevelModel<List<T>>());
		}		
	}
	
	public SequenceModel(Corpus<T> corpus, int min, int max)
		throws IOException
	{
		this(min, max);
		add(corpus);
	}
	
	public void add(Corpus<T> corpus)
		throws IOException
	{
		for(T token : corpus)
			add(token);
	}
	
	public void add(T token)
	{
		alphabet.add(token);
		
		// * Fill the buffers
		for(int i = 0; i < buffers.size(); i++)
		{
			List<T> buffer = buffers.get(i);
			
			buffer.add(token);              // push
			while(buffer.size() > i + min)  // pop
				buffer.remove(0);
		}
		
		// * Add the n-grams to the models
		List<T> superToken = new ArrayList<T>();
		for(int i = 0; i < models.size(); i++)
		{
			superToken = new ArrayList<>(i + min); // do not use clear();
			superToken.addAll(buffers.get(i));
						
			if(superToken.size() == min + i)
				models.get(i).add(superToken);
		}		
	}
	
	public LevelModel<List<T>> model(int length)
	{
		assert(min <= length && length <= max);
		
		return models.get(length - min);
	}

	/**
	 * The shorter-by-one subtokens of this token
	 * @param token
	 * @return
	 */
	public List<List<T>> parents(List<T> token)
	{
		List<List<T>> parents = new ArrayList<List<T>>();
		
		parents.add(token.subList(0, token.size()-1));
		parents.add(token.subList(1, token.size()));		
		
		return parents;
	}
	
	/**
	 * The longer-by-one supertokens of this token
	 * @param token
	 * @return
	 */	
	public List<List<T>> children(List<T> token)
	{
		List<List<T>> children = new ArrayList<List<T>>(alphabet.size() * 2);
		
		List<T> child;
		for(T t : alphabet)
		{
			child = new ArrayList<T>(token.size() + 1);
			child.add(t);
			child.addAll(token);			
			children.add(child);

			child = new ArrayList<T>(token.size() + 1);
			child.addAll(token);			
			child.add(t);
			children.add(child);
		}
		
		return children;
	}

	/**
	 * Relevance of this token according to the relevant model. 
	 * 
	 * Note that in the sequence model NaN relevances are mapped to negative infinity 
	 * (since we are likely only interested in high-relevance tokens in this 
	 * model)
	 * 
	 * @param token
	 * @return
	 */
	public double relevance(List<T> token)
	{
		LevelModel<List<T>> model = model(token.size());
		double r =  model.relevance(token);
		
		if(Double.isNaN(r))
			return Double.NEGATIVE_INFINITY;
		
		return r;
	}
	
	/**
	 * Sorts tokens by their c-relevance
	 * @author peter
	 *
	 * @param <T>
	 */
	public class RelevanceComparator implements Comparator<List<T>>
	{
		@Override
		public int compare(List<T> first, List<T> second) {
			return Double.compare(relevance(first), relevance(second));
		}
	}

	public double frequency(List<T> token)
	{
		LevelModel<List<T>> model = model(token.size());
		return model.frequency(token);
	}

	public double sigNorm(List<T> token)
	{
		LevelModel<List<T>> model = model(token.size());
		return model.relevanceSigNorm(token);
	}

}
