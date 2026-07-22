using System.Text.Json;

namespace LLM.Tokenizer;

/// <summary>
/// Simplest possible tokenizer: one token per Unicode character. Vocab is just the
/// distinct characters seen in the training corpus, sorted for a deterministic mapping.
/// Good enough for a small educational model; swap for a BPE tokenizer later if you
/// want a smaller sequence-length-per-token ratio.
/// </summary>
public sealed class CharTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _charToId;
    private readonly char[] _idToChar;

    public int VocabSize => _idToChar.Length;

    private CharTokenizer(char[] idToChar)
    {
        _idToChar = idToChar;
        _charToId = new Dictionary<char, int>(idToChar.Length);
        for (int i = 0; i < idToChar.Length; i++) _charToId[idToChar[i]] = i;
    }

    public static CharTokenizer BuildFromCorpus(string text)
    {
        var distinct = text.Distinct().Order().ToArray();
        if (distinct.Length == 0)
            throw new ArgumentException("Corpus is empty; cannot build a vocabulary.");
        return new CharTokenizer(distinct);
    }

    public int[] Encode(string text)
    {
        var ids = new int[text.Length];
        for (int i = 0; i < text.Length; i++)
        {
            if (!_charToId.TryGetValue(text[i], out var id))
                throw new ArgumentException(
                    $"Character '{text[i]}' (U+{(int)text[i]:X4}) is not in the tokenizer vocabulary.");
            ids[i] = id;
        }
        return ids;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var chars = ids.Select(id => _idToChar[id]).ToArray();
        return new string(chars);
    }

    private sealed class VocabDto
    {
        public string Chars { get; set; } = "";
    }

    public void SaveVocab(string path)
    {
        var dto = new VocabDto { Chars = new string(_idToChar) };
        File.WriteAllText(path, JsonSerializer.Serialize(dto));
    }

    public static CharTokenizer LoadVocab(string path)
    {
        var json = File.ReadAllText(path);
        var dto = JsonSerializer.Deserialize<VocabDto>(json)
                  ?? throw new InvalidDataException($"Could not parse tokenizer vocab file: {path}");
        return new CharTokenizer(dto.Chars.ToCharArray());
    }
}
