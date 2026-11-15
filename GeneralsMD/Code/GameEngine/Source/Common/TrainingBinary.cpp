// GeneralsAI training export: binary observation writer. See TrainingBinary.h.
// This file deliberately includes no engine headers: zlib's Z_PREFIX macros
// (Byte, uLong, ...) must not leak into engine code.
#include "Common/TrainingBinary.h"

#include <float.h>
#include <string.h>
#include <stdexcept>
#include <zlib.h>

namespace training_binary {

namespace {
const Tag TAG_NULL = 0, TAG_FALSE = 1, TAG_TRUE = 2, TAG_INT = 3, TAG_UINT = 4, TAG_REAL = 5,
	TAG_STRING = 6, TAG_ARRAY = 7, TAG_OBJECT = 8;

Bits packed(unsigned high, unsigned low)
{
	return (static_cast<Bits>(high) << 32) | low;
}

Bits fnvOffset()
{
	// 1469598103934665603: the 64-bit FNV-1a offset basis.
	return (static_cast<Bits>(0xcbf29ce4u) << 32) | 0x84222325u;
}

Bits fnvPrime()
{
	// 1099511628211: the 64-bit FNV-1a prime.
	return (static_cast<Bits>(0x00000100u) << 32) | 0x000001b3u;
}

// Length of a valid UTF-8 sequence starting at text[i], or 0.
size_t utf8Length(const unsigned char *text, size_t i, size_t length)
{
	const unsigned char c = text[i];
	size_t count;
	unsigned minimum;
	unsigned value;
	if (c < 0x80) return 1;
	if ((c & 0xe0) == 0xc0) { count = 2; minimum = 0x80; value = c & 0x1f; }
	else if ((c & 0xf0) == 0xe0) { count = 3; minimum = 0x800; value = c & 0x0f; }
	else if ((c & 0xf8) == 0xf0) { count = 4; minimum = 0x10000; value = c & 0x07; }
	else return 0;
	if (i + count > length) return 0;
	for (size_t k = 1; k < count; ++k) {
		if ((text[i + k] & 0xc0) != 0x80) return 0;
		value = (value << 6) | (text[i + k] & 0x3f);
	}
	if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return 0;
	return count;
}
} // namespace

unsigned Buffer::reserve(size_t count)
{
	if (count > (512u * 1024u * 1024u / 9u) - values.size())
		throw std::runtime_error("Binary observation exceeds block memory budget");
	const unsigned start = static_cast<unsigned>(values.size());
	values.resize(values.size() + count);
	tags.resize(values.size());
	return start;
}

unsigned Cursor::next()
{
	if (current == end) throw std::runtime_error("Too many direct observation fields");
	return current++;
}

void Cursor::finish() const
{
	if (current != end) throw std::runtime_error("Missing direct observation fields");
}

Writer::Writer(FILE *file) : m_file(file), m_literal(1024)
{
	if (sizeof(double) != 8) throw std::runtime_error("IEEE doubles required");
	const unsigned short endian = 1;
	if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
		throw std::runtime_error("Binary writer requires little-endian host");
	if (fwrite("GOBS1\r\n\032", 1, 8, m_file) != 8) throw std::runtime_error("Cannot write binary training export");
}

unsigned Writer::intern(const char *text)
{
	const std::string key(text);
	std::map<std::string, unsigned>::const_iterator found = m_strings.find(key);
	if (found != m_strings.end()) return found->second;
	const unsigned id = static_cast<unsigned>(m_strings.size());
	m_strings[key] = id;
	m_newStrings.push_back(key);
	return id;
}

unsigned Writer::finishSchema()
{
	Bits hash = fnvOffset();
	for (size_t i = 0; i < m_keys.size(); ++i) hash = (hash ^ m_keys[i]) * fnvPrime();
	std::vector<unsigned> &candidates = m_layouts[hash];
	for (size_t c = 0; c < candidates.size(); ++c)
		if (m_schemas[candidates[c]] == m_keys) return candidates[c];
	const unsigned id = static_cast<unsigned>(m_schemas.size());
	m_schemas.push_back(m_keys);
	candidates.push_back(id);
	m_newSchemas.push_back(m_keys);
	return id;
}

void Writer::begin(Buffer &buffer)
{
	buffer.clear();
	buffer.reserve(1);
}

Cursor Writer::object(Buffer &buffer, unsigned node, const char *const *fields, size_t count)
{
	size_t hash = count;
	for (size_t i = 0; i < count; ++i) hash = (hash * 31) ^ (reinterpret_cast<size_t>(fields[i]) >> 2);
	LiteralLayout &known = m_literal[(hash ^ (hash >> 10)) & 1023];
	unsigned layout = known.layout;
	bool same = known.fields.size() == count;
	for (size_t k = 0; same && k < count; ++k) same = known.fields[k] == fields[k];
	if (!same) {
		m_keys.clear();
		for (size_t j = 0; j < count; ++j) m_keys.push_back(intern(fields[j]));
		layout = finishSchema();
		known.fields.assign(fields, fields + count);
		known.layout = layout;
	}
	const unsigned first = buffer.reserve(count);
	buffer.tags[node] = TAG_OBJECT;
	buffer.values[node] = packed(layout, first);
	Cursor cursor = {first, first + static_cast<unsigned>(count)};
	return cursor;
}

Cursor Writer::namedObject(Buffer &buffer, unsigned node, const std::vector<std::string> &fields)
{
	m_keys.clear();
	for (size_t i = 0; i < fields.size(); ++i) m_keys.push_back(intern(fields[i].c_str()));
	const unsigned layout = finishSchema();
	const unsigned first = buffer.reserve(fields.size());
	buffer.tags[node] = TAG_OBJECT;
	buffer.values[node] = packed(layout, first);
	Cursor cursor = {first, first + static_cast<unsigned>(fields.size())};
	return cursor;
}

Cursor Writer::array(Buffer &buffer, unsigned node, size_t count)
{
	const unsigned first = buffer.reserve(count);
	buffer.tags[node] = TAG_ARRAY;
	buffer.values[node] = packed(static_cast<unsigned>(count), first);
	Cursor cursor = {first, first + static_cast<unsigned>(count)};
	return cursor;
}

void Writer::putBool(Buffer &buffer, unsigned node, bool value)
{
	buffer.tags[node] = value ? TAG_TRUE : TAG_FALSE;
	buffer.values[node] = 0;
}

void Writer::putInt(Buffer &buffer, unsigned node, __int64 value)
{
	buffer.tags[node] = TAG_INT;
	buffer.values[node] = static_cast<Bits>(value);
}

void Writer::putUInt(Buffer &buffer, unsigned node, unsigned __int64 value)
{
	buffer.tags[node] = TAG_UINT;
	buffer.values[node] = value;
}

void Writer::putReal(Buffer &buffer, unsigned node, double value)
{
	Bits bits = 0;
	// Nonfinite values are null, like the JSON exporter.
	if (_finite(value)) {
		memcpy(&bits, &value, 8);
		buffer.tags[node] = TAG_REAL;
	} else buffer.tags[node] = TAG_NULL;
	buffer.values[node] = bits;
}

void Writer::putString(Buffer &buffer, unsigned node, const char *value)
{
	buffer.tags[node] = TAG_STRING;
	buffer.values[node] = intern(value);
}

void Writer::putNull(Buffer &buffer, unsigned node)
{
	buffer.tags[node] = TAG_NULL;
	buffer.values[node] = 0;
}

void Writer::u32(unsigned value)
{
	const unsigned char bytes[4] = {static_cast<unsigned char>(value), static_cast<unsigned char>(value >> 8),
		static_cast<unsigned char>(value >> 16), static_cast<unsigned char>(value >> 24)};
	if (fwrite(bytes, 1, 4, m_file) != 4) throw std::runtime_error("Cannot write binary training export");
}

void Writer::write(const std::string &envelope, const Buffer *observation)
{
	// {"record":...,"strings":[...],"schemas":[...]}, usually with empty lists.
	std::string text = "{\"record\":";
	text += envelope;
	text += ",\"strings\":[";
	for (size_t s = 0; s < m_newStrings.size(); ++s) {
		if (s) text += ',';
		appendJsonString(text, m_newStrings[s].c_str(), m_newStrings[s].size());
	}
	text += "],\"schemas\":[";
	char number[16];
	for (size_t i = 0; i < m_newSchemas.size(); ++i) {
		if (i) text += ',';
		text += '[';
		for (size_t k = 0; k < m_newSchemas[i].size(); ++k) {
			if (k) text += ',';
			sprintf(number, "%u", m_newSchemas[i][k]);
			text += number;
		}
		text += ']';
	}
	text += "]}";
	if (text.size() > 16u * 1024u * 1024u) throw std::runtime_error("Binary metadata exceeds block budget");
	const size_t count = observation ? observation->tags.size() : 0;
	m_raw.resize(count * 9);
	if (count) {
		memcpy(&m_raw[0], &observation->tags[0], count);
		memcpy(&m_raw[count], &observation->values[0], count * 8);
	}
	// zlib 1.1.4 has no compressBound: its documented bound is 0.1% + 12 bytes.
	uLongf size = m_raw.empty() ? 0 : static_cast<uLongf>(m_raw.size() + m_raw.size() / 1000 + 64);
	m_compressed.resize(size ? size : 1);
	if (!m_raw.empty() && compress2(&m_compressed[0], &size, &m_raw[0], static_cast<uLong>(m_raw.size()), 1) != Z_OK)
		throw std::runtime_error("Cannot compress binary observation");
	const bool useCompression = !m_raw.empty() && size < m_raw.size();
	const unsigned char *body = m_raw.empty() ? 0 : (useCompression ? &m_compressed[0] : &m_raw[0]);
	const unsigned bodySize = useCompression ? static_cast<unsigned>(size) : static_cast<unsigned>(m_raw.size());
	uLong crc = crc32(0L, reinterpret_cast<const Bytef *>(text.data()), static_cast<uInt>(text.size()));
	if (!m_raw.empty()) crc = crc32(crc, &m_raw[0], static_cast<uInt>(m_raw.size()));
	if (fwrite("BLK1", 1, 4, m_file) != 4) throw std::runtime_error("Cannot write binary training export");
	u32(static_cast<unsigned>(text.size())); u32(static_cast<unsigned>(count));
	u32(static_cast<unsigned>(m_raw.size())); u32(bodySize); u32(static_cast<unsigned>(crc)); u32(useCompression ? 1 : 0);
	if (fwrite(text.data(), 1, text.size(), m_file) != text.size()
		|| (bodySize && fwrite(body, 1, bodySize, m_file) != bodySize))
		throw std::runtime_error("Cannot write binary training export");
	m_newStrings.clear();
	m_newSchemas.clear();
}

void appendJsonString(std::string &out, const char *text, size_t length)
{
	static const char hex[] = "0123456789abcdef";
	const unsigned char *bytes = reinterpret_cast<const unsigned char *>(text);
	out += '"';
	size_t i = 0;
	while (i < length) {
		const unsigned char c = bytes[i];
		if (c == '"') { out += "\\\""; ++i; }
		else if (c == '\\') { out += "\\\\"; ++i; }
		else if (c == '\n') { out += "\\n"; ++i; }
		else if (c == '\r') { out += "\\r"; ++i; }
		else if (c == '\t') { out += "\\t"; ++i; }
		else if (c == '\b') { out += "\\b"; ++i; }
		else if (c == '\f') { out += "\\f"; ++i; }
		else if (c < 0x20) {
			out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; ++i;
		} else if (c < 0x80) { out += static_cast<char>(c); ++i; }
		else {
			// Invalid UTF-8 (such as legacy code-page names) becomes U+FFFD.
			const size_t n = utf8Length(bytes, i, length);
			if (n) { out.append(text + i, n); i += n; }
			else { out += "\xef\xbf\xbd"; ++i; }
		}
	}
	out += '"';
}

void appendJsonNumber(std::string &out, double value)
{
	if (!_finite(value)) { out += "null"; return; }
	char text[40];
	sprintf(text, "%.17g", value);
	out += text;
	// Keep floating-point values recognisable as floats, like the JSON exporter.
	if (!strpbrk(text, ".eEn")) out += ".0";
}

} // namespace training_binary
