// GeneralsAI training export: a minimal ordered JSON value (C++98, VC6-compatible).
// Replaces nlohmann::ordered_json for the record envelopes and command receipts.
// Integers keep their signedness, like nlohmann, so the binary encoder writes the
// same tags as the Generals Online exporter.
#pragma once

#include <string>
#include <vector>

namespace training_binary { class Writer; struct Buffer; }

class TrainingJson
{
public:
	enum Type { NUL, BOOLEAN, INTEGER, UNSIGNED, REAL, STRING, ARRAY, OBJECT };

	TrainingJson() : m_type(NUL), m_bool(false), m_int(0), m_uint(0), m_real(0) {}
	TrainingJson(const TrainingJson &other);
	~TrainingJson();
	TrainingJson &operator=(const TrainingJson &other);

	static TrainingJson boolean(bool value);
	static TrainingJson integer(__int64 value);
	static TrainingJson unsignedInteger(unsigned __int64 value);
	static TrainingJson real(double value);
	static TrainingJson string(const char *value);
	static TrainingJson string(const std::string &value);
	static TrainingJson array();
	static TrainingJson object();

	Type type() const { return m_type; }
	bool isNull() const { return m_type == NUL; }
	bool isObject() const { return m_type == OBJECT; }
	bool isArray() const { return m_type == ARRAY; }
	bool isString() const { return m_type == STRING; }
	bool isNumber() const { return m_type == INTEGER || m_type == UNSIGNED || m_type == REAL; }
	bool asBool() const { return m_bool; }
	unsigned asUnsigned() const;
	const std::string &asString() const { return m_string; }

	size_t size() const { return m_items.size(); }
	const TrainingJson &at(size_t index) const { return *m_items[index]; }
	TrainingJson &at(size_t index) { return *m_items[index]; }
	void push(const TrainingJson &value);

	// Objects keep insertion order; set() replaces an existing key in place.
	TrainingJson &set(const char *key, const TrainingJson &value);
	// For keys known to be new (large generated objects): no duplicate search.
	TrainingJson &append(const char *key, const TrainingJson &value);
	const TrainingJson *find(const char *key) const;
	TrainingJson *find(const char *key);
	bool contains(const char *key) const { return find(key) != 0; }
	void erase(const char *key);
	const std::string &keyAt(size_t index) const { return m_keys[index]; }

	void dump(std::string &out) const;
	std::string dump() const { std::string out; dump(out); return out; }
	// Encode into an observation buffer node (the binary writer's tag/value columns).
	void encode(training_binary::Writer &writer, training_binary::Buffer &buffer, unsigned node) const;

private:
	Type m_type;
	bool m_bool;
	__int64 m_int;
	unsigned __int64 m_uint;
	double m_real;
	std::string m_string;
	std::vector<std::string> m_keys;
	std::vector<TrainingJson *> m_items;
	void clearItems();
};
