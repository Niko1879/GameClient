// GeneralsAI training export: binary observation writer (C++98, VC6-compatible).
// Byte-compatible with the Generals Online exporter's GOBS1 format: versioned,
// independently compressed blocks; field layouts and strings are interned across
// the file; scalar tags and 64-bit values are columns; the small record envelope
// is JSON text. zlib stays in TrainingBinary.cpp (its Z_PREFIX macros clash with
// engine typedefs).
#pragma once

#include <stdio.h>
#include <map>
#include <string>
#include <vector>

namespace training_binary {

typedef unsigned char Tag;
typedef unsigned __int64 Bits;

struct Buffer {
	std::vector<Tag> tags;
	std::vector<Bits> values;
	unsigned reserve(size_t count);
	void clear() { tags.clear(); values.clear(); }
};

struct Cursor {
	unsigned current, end;
	unsigned next();
	void finish() const;
};

class Writer {
public:
	explicit Writer(FILE *file);
	void begin(Buffer &buffer);
	// `fields` must be an array of string literals: their addresses identify the layout.
	Cursor object(Buffer &buffer, unsigned node, const char *const *fields, size_t count);
	// Fields with non-literal names (copied JSON objects).
	Cursor namedObject(Buffer &buffer, unsigned node, const std::vector<std::string> &fields);
	Cursor array(Buffer &buffer, unsigned node, size_t count);
	void putBool(Buffer &buffer, unsigned node, bool value);
	void putInt(Buffer &buffer, unsigned node, __int64 value);
	void putUInt(Buffer &buffer, unsigned node, unsigned __int64 value);
	void putReal(Buffer &buffer, unsigned node, double value);
	void putString(Buffer &buffer, unsigned node, const char *value);
	void putNull(Buffer &buffer, unsigned node);
	// One record: the JSON envelope text plus an optional observation buffer.
	void write(const std::string &envelope, const Buffer *observation);

private:
	struct LiteralLayout {
		std::vector<const char *> fields;
		unsigned layout;
		LiteralLayout() : layout(0) {}
	};
	FILE *m_file;
	std::map<std::string, unsigned> m_strings;
	std::map<Bits, std::vector<unsigned> > m_layouts;
	std::vector<std::vector<unsigned> > m_schemas;
	std::vector<unsigned> m_keys;
	std::vector<LiteralLayout> m_literal;
	std::vector<std::string> m_newStrings;
	std::vector<std::vector<unsigned> > m_newSchemas;
	std::vector<unsigned char> m_raw, m_compressed;

	unsigned intern(const char *text);
	unsigned finishSchema();
	void u32(unsigned value);
};

// JSON text helpers shared with the envelope builder.
void appendJsonString(std::string &out, const char *text, size_t length);
void appendJsonNumber(std::string &out, double value);

} // namespace training_binary
