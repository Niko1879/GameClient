// GeneralsAI training export: minimal ordered JSON value. See TrainingJson.h.
#include "Common/TrainingJson.h"
#include "Common/TrainingBinary.h"

#include <stdio.h>
#include <string.h>

TrainingJson::TrainingJson(const TrainingJson &other)
	: m_type(other.m_type), m_bool(other.m_bool), m_int(other.m_int), m_uint(other.m_uint),
	  m_real(other.m_real), m_string(other.m_string), m_keys(other.m_keys)
{
	for (size_t i = 0; i < other.m_items.size(); ++i) m_items.push_back(new TrainingJson(*other.m_items[i]));
}

TrainingJson::~TrainingJson()
{
	clearItems();
}

void TrainingJson::clearItems()
{
	for (size_t i = 0; i < m_items.size(); ++i) delete m_items[i];
	m_items.clear();
}

TrainingJson &TrainingJson::operator=(const TrainingJson &other)
{
	if (this == &other) return *this;
	TrainingJson copy(other);
	clearItems();
	m_type = copy.m_type; m_bool = copy.m_bool; m_int = copy.m_int; m_uint = copy.m_uint;
	m_real = copy.m_real; m_string.swap(copy.m_string); m_keys.swap(copy.m_keys); m_items.swap(copy.m_items);
	return *this;
}

TrainingJson TrainingJson::boolean(bool value) { TrainingJson j; j.m_type = BOOLEAN; j.m_bool = value; return j; }
TrainingJson TrainingJson::integer(__int64 value) { TrainingJson j; j.m_type = INTEGER; j.m_int = value; return j; }
TrainingJson TrainingJson::unsignedInteger(unsigned __int64 value) { TrainingJson j; j.m_type = UNSIGNED; j.m_uint = value; return j; }
TrainingJson TrainingJson::real(double value) { TrainingJson j; j.m_type = REAL; j.m_real = value; return j; }
TrainingJson TrainingJson::string(const char *value) { TrainingJson j; j.m_type = STRING; j.m_string = value; return j; }
TrainingJson TrainingJson::string(const std::string &value) { TrainingJson j; j.m_type = STRING; j.m_string = value; return j; }
TrainingJson TrainingJson::array() { TrainingJson j; j.m_type = ARRAY; return j; }
TrainingJson TrainingJson::object() { TrainingJson j; j.m_type = OBJECT; return j; }

unsigned TrainingJson::asUnsigned() const
{
	if (m_type == UNSIGNED) return static_cast<unsigned>(m_uint);
	if (m_type == INTEGER) return static_cast<unsigned>(m_int);
	if (m_type == REAL) return static_cast<unsigned>(m_real);
	return 0;
}

void TrainingJson::push(const TrainingJson &value)
{
	if (m_type != ARRAY) { clearItems(); m_keys.clear(); m_type = ARRAY; }
	m_items.push_back(new TrainingJson(value));
}

TrainingJson &TrainingJson::set(const char *key, const TrainingJson &value)
{
	if (m_type != OBJECT) { clearItems(); m_keys.clear(); m_type = OBJECT; }
	for (size_t i = 0; i < m_keys.size(); ++i)
		if (m_keys[i] == key) { *m_items[i] = value; return *m_items[i]; }
	m_keys.push_back(key);
	m_items.push_back(new TrainingJson(value));
	return *m_items.back();
}

TrainingJson &TrainingJson::append(const char *key, const TrainingJson &value)
{
	if (m_type != OBJECT) { clearItems(); m_keys.clear(); m_type = OBJECT; }
	m_keys.push_back(key);
	m_items.push_back(new TrainingJson(value));
	return *m_items.back();
}

const TrainingJson *TrainingJson::find(const char *key) const
{
	if (m_type != OBJECT) return 0;
	for (size_t i = 0; i < m_keys.size(); ++i) if (m_keys[i] == key) return m_items[i];
	return 0;
}

TrainingJson *TrainingJson::find(const char *key)
{
	if (m_type != OBJECT) return 0;
	for (size_t i = 0; i < m_keys.size(); ++i) if (m_keys[i] == key) return m_items[i];
	return 0;
}

void TrainingJson::erase(const char *key)
{
	if (m_type != OBJECT) return;
	for (size_t i = 0; i < m_keys.size(); ++i)
		if (m_keys[i] == key) {
			delete m_items[i];
			m_items.erase(m_items.begin() + i);
			m_keys.erase(m_keys.begin() + i);
			return;
		}
}

void TrainingJson::dump(std::string &out) const
{
	char text[32];
	switch (m_type) {
	case NUL: out += "null"; break;
	case BOOLEAN: out += m_bool ? "true" : "false"; break;
	case INTEGER: sprintf(text, "%I64d", m_int); out += text; break;
	case UNSIGNED: sprintf(text, "%I64u", m_uint); out += text; break;
	case REAL: training_binary::appendJsonNumber(out, m_real); break;
	case STRING: training_binary::appendJsonString(out, m_string.data(), m_string.size()); break;
	case ARRAY: {
		out += '[';
		for (size_t i = 0; i < m_items.size(); ++i) {
			if (i) out += ',';
			m_items[i]->dump(out);
		}
		out += ']';
		break;
	}
	case OBJECT: {
		out += '{';
		for (size_t k = 0; k < m_items.size(); ++k) {
			if (k) out += ',';
			training_binary::appendJsonString(out, m_keys[k].data(), m_keys[k].size());
			out += ':';
			m_items[k]->dump(out);
		}
		out += '}';
		break;
	}
	}
}

void TrainingJson::encode(training_binary::Writer &writer, training_binary::Buffer &buffer, unsigned node) const
{
	switch (m_type) {
	case NUL: writer.putNull(buffer, node); break;
	case BOOLEAN: writer.putBool(buffer, node, m_bool); break;
	case INTEGER: writer.putInt(buffer, node, m_int); break;
	case UNSIGNED: writer.putUInt(buffer, node, m_uint); break;
	case REAL: writer.putReal(buffer, node, m_real); break;
	case STRING: writer.putString(buffer, node, m_string.c_str()); break;
	case ARRAY: {
		training_binary::Cursor items = writer.array(buffer, node, m_items.size());
		for (size_t i = 0; i < m_items.size(); ++i) m_items[i]->encode(writer, buffer, items.next());
		break;
	}
	case OBJECT: {
		training_binary::Cursor fields = writer.namedObject(buffer, node, m_keys);
		for (size_t k = 0; k < m_items.size(); ++k) m_items[k]->encode(writer, buffer, fields.next());
		break;
	}
	}
}
