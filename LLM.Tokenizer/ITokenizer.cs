namespace LLM.Tokenizer;

public interface ITokenizer
{
    int VocabSize { get; }
    int[] Encode(string text);
    string Decode(IEnumerable<int> ids);
}
