/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/object.h"
#include "ns3/uinteger.h"
#include "ns3/traced-value.h"
#include "ns3/trace-source-accessor.h"

#include "ns3/core-module.h"

#include <iostream>

using namespace ns3;

// Probably should avoid using ns std;
using namespace std;

class TraceTest : public Object
{
public:

    static TypeId GetTypeId (void)
    {
        static TypeId tid = TypeId("TraceTest")
        .SetParent(Object::GetTypeId())
        .SetGroupName("Tracing")
        .AddConstructor<TraceTest> ()
        .AddTraceSource("Integer value",
                        "An integer value to trace.",
                        MakeTraceSourceAccessor(&TraceTest::val_Int),
                        "ns3::TracedValueCallback::Int32")
        ;
        return tid;
    }

    TraceTest() {}
    TracedValue<int32_t> val_Int;
};

void IntTrace (int32_t oldValue, int32_t newValue)
{
    std::cout<<"Traced "<<oldValue<<" to "<<newValue<<std::endl;
}

NS_LOG_COMPONENT_DEFINE("TracingTest");

int main(int argc, char *argv[])
{
    std::cout<<"Tracing test start, piped from std out...."<<std::endl;
    NS_LOG_INFO("Tracing test start, piped from NS_LOG_INFO");

    // $ export NS_LOG = TracingTest=info

    Ptr<TraceTest> traceTest = CreateObject<TraceTest> ();
    traceTest->TraceConnectWithoutContext("Integer value", MakeCallback(&IntTrace));

    traceTest->val_Int = 1000;
    return 0;
}


