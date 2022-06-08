/* Tags for tracking simulation info.
*/
#ifndef EXPERIMENT_TAGS_H
#define EXPERIMENT_TAGS_H

#include "ns3/core-module.h"
#include "ns3/network-module.h"

using namespace ns3;

// A timestamp tag that can be added to a packet.
class TimestampTag : public Tag
{
public:
    static TypeId GetTypeId(void)
    {
        static TypeId tid = TypeId("ns3::TimestampTag")
                                .SetParent<Tag>()
                                .AddConstructor<TimestampTag>()
                                .AddAttribute("Timestamp",
                                              "Timestamp to save in tag.",
                                              EmptyAttributeValue(),
                                              MakeTimeAccessor(&TimestampTag::timestamp),
                                              MakeTimeChecker());
        return tid;
    };
    TypeId GetInstanceTypeId(void) const { return GetTypeId(); };
    uint32_t GetSerializedSize(void) const { return sizeof(timestamp); };
    void Serialize(TagBuffer i) const
    {
        i.Write(reinterpret_cast<const uint8_t *>(&timestamp),
                sizeof(timestamp));
    };
    void Deserialize(TagBuffer i)
    {
        i.Read(reinterpret_cast<uint8_t *>(&timestamp), sizeof(timestamp));
    };
    void Print(std::ostream &os) const
    {
        os << "t=" << timestamp;
    };

    // these are our accessors to our tag structure
    void SetTime(Time time) { timestamp = time; };
    Time GetTime() { return timestamp; };

private:
    Time timestamp;
};

// A tag with two integer values for workload and application ids.
class IdTag : public Tag
{
public:
    static TypeId GetTypeId(void)
    {
        static TypeId tid =
            TypeId("ns3::IntTag")
                .SetParent<Tag>()
                .AddConstructor<IdTag>()
                .AddAttribute("workload",
                              "Workload id to save in tag.",
                              EmptyAttributeValue(),
                              MakeUintegerAccessor(&IdTag::workload),
                              MakeUintegerChecker<u_int32_t>())
                .AddAttribute("application",
                              "Application id to save in tag.",
                              EmptyAttributeValue(),
                              MakeUintegerAccessor(&IdTag::application),
                              MakeUintegerChecker<u_int32_t>());
        return tid;
    };
    TypeId GetInstanceTypeId(void) const { return GetTypeId(); };
    uint32_t GetSerializedSize(void) const
    {
        return sizeof(workload) + sizeof(application);
    };
    void Serialize(TagBuffer i) const
    {
        i.Write(reinterpret_cast<const uint8_t *>(&workload),
                sizeof(workload));
        i.Write(reinterpret_cast<const uint8_t *>(&application),
                sizeof(application));
    };
    void Deserialize(TagBuffer i)
    {
        i.Read(reinterpret_cast<uint8_t *>(&workload), sizeof(workload));
        i.Read(reinterpret_cast<uint8_t *>(&application), sizeof(application));
    };
    void Print(std::ostream &os) const
    {
        os << "w=" << workload << ", "
           << "a=" << application;
    };

    // these are our accessors to our tag structure
    void SetWorkload(u_int32_t newval) { workload = newval; };
    u_int32_t GetWorkload() { return workload; };
    void SetApplication(u_int32_t newval) { application = newval; };
    u_int32_t GetApplication() { return application; };

private:
    u_int32_t workload;
    u_int32_t application;
};

// A tag to check message ids and see if they are preserved across fragments
class MessageTag : public Tag
{
    public:
        
    static TypeId GetTypeId (void)
    {
    static TypeId tid = TypeId ("ns3::MessageTag")
        .SetParent<Tag> ()
        .AddConstructor<MessageTag> ()
        .AddAttribute ("SimpleValue",
                    "A simple value",
                    EmptyAttributeValue (),
                    MakeUintegerAccessor (&MessageTag::m_simpleValue),
                    MakeUintegerChecker<uint8_t> ());
    return tid;
    }
    TypeId GetInstanceTypeId(void) const
    {
        return GetTypeId();
    }
    uint32_t GetSerializedSize(void) const
    {
        return sizeof(m_simpleValue);
    }
    void Serialize(TagBuffer i) const
    {
        i.Write(reinterpret_cast<const uint8_t *>(&m_simpleValue),
                sizeof(m_simpleValue));
    }
    void Deserialize(TagBuffer i)
    {
        i.Read(reinterpret_cast<uint8_t *>(&m_simpleValue),
                sizeof(m_simpleValue));
    }
    void Print(std::ostream &os) const
    {
        os << "v=" << (uint32_t)m_simpleValue;
    }
    void SetSimpleValue(uint32_t value)
    {
        m_simpleValue = value;
    }
    uint32_t GetSimpleValue(void) const
    {
        return m_simpleValue;
    }

    private:
        uint32_t m_simpleValue;  
};

#endif // EXPERIMENT_TAGS_H
