import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sentence_transformers import SentenceTransformer, util

# Lazy load to avoid loading during simple imports
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def similarity_check(orig: str, modified: str, min_sim=0.75) -> bool:
    """
    Checks if the modified email preserves the semantic meaning of the original.
    Returns True if similarity >= min_sim, else False.
    """
    if orig == modified:
        return True
        
    # We only take the first 500 characters to speed up encoding
    orig_trunc = orig[:500]
    mod_trunc = modified[:500]
    
    model = get_model()
    emb = model.encode([orig_trunc, mod_trunc], convert_to_tensor=True)
    sim = float(util.cos_sim(emb[0], emb[1])[0][0])
    
    return sim >= min_sim

def get_similarity_score(orig: str, modified: str) -> float:
    if orig == modified:
        return 1.0
    orig_trunc = orig[:500]
    mod_trunc = modified[:500]
    model = get_model()
    emb = model.encode([orig_trunc, mod_trunc], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1])[0][0])
