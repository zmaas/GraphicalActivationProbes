import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Setup OpenAI client - will use API key from environment variables
# Priority: 
# 1. Explicitly set OPENAI_API_KEY from .env 
# 2. Default OpenAI environment variable lookup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_examples(
    prompt: str,
    num_examples: int = 30,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    cache_dir: str = "cache/synthetic_data",
    cache_file: Optional[str] = None,
    force_regenerate: bool = False
) -> List[str]:
    """
    Generate text examples using OpenAI API.
    
    Args:
        prompt: Prompt specifying what kind of examples to generate
        num_examples: Number of examples to generate
        model: OpenAI model to use 
        temperature: Sampling temperature (higher = more random)
        cache_dir: Directory to cache generated examples
        cache_file: Specific cache file to use (if None, will be derived from prompt)
        force_regenerate: If True, ignores cache and regenerates examples
        
    Returns:
        List of generated text examples
    """
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or add it to your .env file."
        )
        
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename from prompt if not provided
    if cache_file is None:
        # Create a deterministic but short identifier from the prompt
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:10]
        cache_file = f"{prompt_hash}_{num_examples}_{model.replace('/', '_')}.json"
    
    cache_path = os.path.join(cache_dir, cache_file)
    
    # Check if cached results exist and return them
    if not force_regenerate and os.path.exists(cache_path):
        print(f"Loading cached examples from {cache_path}")
        with open(cache_path, 'r') as f:
            cached_data: List[str] = json.load(f)
            return cached_data
    
    # System message instructing the model what to do
    system_message = """You are a helpful assistant that generates text examples based on the user's request.
Each example should be a complete, self-contained text. The examples should be diverse and realistic.
Provide ONLY the example texts, with each example separated by a ||| delimiter.
Do not include any explanations, numbering, or other text."""

    # Generate examples in batches
    batch_size = 5  # Number of examples to generate per API call
    batches = (num_examples + batch_size - 1) // batch_size  # Ceiling division
    
    all_examples: List[str] = []
    
    for i in tqdm(range(batches), desc="Generating examples"):
        # Calculate how many examples to generate in this batch
        remaining = num_examples - len(all_examples)
        current_batch_size = min(batch_size, remaining)
        
        batch_prompt = f"{prompt}\n\nGenerate {current_batch_size} examples, separated by '|||'."
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=temperature,
            )
            
            # Extract examples from the response
            text = response.choices[0].message.content
            if text:
                batch_examples = [ex.strip() for ex in text.split("|||") if ex.strip()]
                
                # Add examples to our collection
                all_examples.extend(batch_examples)
            
        except Exception as e:
            print(f"Error generating examples: {e}")
            # Continue with what we have so far
    
    # Trim to exactly the requested number
    all_examples = all_examples[:num_examples]
    
    # Cache the results
    with open(cache_path, 'w') as f:
        json.dump(all_examples, f)
    
    return all_examples

def create_sentiment_dataset(
    topic: str,
    num_positive: int = 30,
    num_negative: int = 30,
    model: str = "gpt-4o",
    cache_dir: str = "cache/synthetic_data",
    force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Create a balanced sentiment dataset on a specific topic.
    
    Args:
        topic: The topic to generate examples about
        num_positive: Number of positive sentiment examples
        num_negative: Number of negative sentiment examples
        model: OpenAI model to use
        cache_dir: Directory to cache generated examples
        force_regenerate: If True, ignores cache and regenerates examples
        
    Returns:
        DataFrame with 'text' and 'label' columns (1=positive, 0=negative)
    """
    # Generate positive examples
    positive_prompt = (
        f"Generate diverse, realistic examples of positive sentiment text about {topic}. "
        f"Each example should express clear positive sentiment and should be about {topic}. "
        f"Examples should be 2-4 sentences long."
    )

    positive_examples = generate_examples(
        prompt=positive_prompt,
        num_examples=num_positive,
        model=model,
        cache_dir=cache_dir,
        cache_file=f"{topic}_positive_{num_positive}_{model.replace('/', '_')}.json",
        force_regenerate=force_regenerate
    )
    
    # Generate negative examples
    negative_prompt = (
        f"Generate diverse, realistic examples of negative sentiment text about {topic}. "
        f"Each example should express clear negative sentiment and should be about {topic}. "
        f"Examples should be 2-4 sentences long."
    )

    negative_examples = generate_examples(
        prompt=negative_prompt,
        num_examples=num_negative,
        model=model,
        cache_dir=cache_dir,
        cache_file=f"{topic}_negative_{num_negative}_{model.replace('/', '_')}.json",
        force_regenerate=force_regenerate
    )
    
    # Create dataframe
    texts = positive_examples + negative_examples
    labels = [1] * len(positive_examples) + [0] * len(negative_examples)
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    return df

def generate_multiclass_dataset(
    topics: List[str],
    num_examples_per_topic: int = 30,
    model: str = "gpt-4o",
    cache_dir: str = "cache/synthetic_data",
    force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Generate a multi-class dataset with different topics.
    
    Args:
        topics: List of topics to generate examples for
        num_examples_per_topic: Number of examples per topic
        model: OpenAI model to use
        cache_dir: Directory to cache generated examples
        force_regenerate: If True, ignores cache and regenerates examples
        
    Returns:
        DataFrame with 'text' and 'topic' columns
    """
    all_texts = []
    all_topics = []
    
    for topic in topics:
        prompt = (
            f"Generate diverse, realistic examples of text about {topic}. "
            f"Each example should clearly be about {topic} and not other topics. "
            f"Examples should be 2-4 sentences long."
        )

        examples = generate_examples(
            prompt=prompt,
            num_examples=num_examples_per_topic,
            model=model,
            cache_dir=cache_dir,
            cache_file=f"{topic}_{num_examples_per_topic}_{model.replace('/', '_')}.json",
            force_regenerate=force_regenerate
        )
        
        all_texts.extend(examples)
        all_topics.extend([topic] * len(examples))
    
    # Create dataframe
    df = pd.DataFrame({
        'text': all_texts,
        'topic': all_topics
    })
    
    return df