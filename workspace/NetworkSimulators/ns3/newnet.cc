/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"

using namespace ns3; 

// Net network topology
//
//       10.1.1.0
// n0 -------------- n1   n2   n3   n4
//    point-to-point  |    |    |    |
//                    ================
//                      LAN 10.1.2.0

NS_LOG_COMPONENT_DEFINE("NewnetTest");

int main(int argc, char *argv[]){

    bool verbose = true;
    uint32_t nCsma = 3;
    uint32_t nPackets = 1;

    CommandLine cmd;
    cmd.AddValue("nCsma", "Number of \"extra\" CSMA nodes/devices", nCsma);
    cmd.AddValue("verbose", "Tell echo applications to log if true", verbose);
    cmd.AddValue("nPackets", "Number of packets to echo", nPackets);


    cmd.Parse(argc, argv);

    if (verbose)
    {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }
    // Sanity check, nCsma >=1 always, nPackets >=1 always
    nCsma = nCsma == 0 ? 1 : nCsma;
    nPackets = nPackets == 0 ? 1 : nPackets;

    // Create the P2P nodes first 
    NodeContainer p2pNodes;
    p2pNodes.Create(2);

    // Create CSMA node containers 
    NodeContainer csmaNodes;
    csmaNodes.Add(p2pNodes.Get (1));
    csmaNodes.Create(nCsma);

    // Bind the P2P devices inside th P2P containers 
    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute("DataRate", StringValue ("5Mbps"));
    pointToPoint.SetChannelAttribute("Delay", StringValue ("2ms"));
    pointToPoint.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue ("50p"));

    NetDeviceContainer p2pDevices;
    p2pDevices = pointToPoint.Install(p2pNodes);

    // Bind the CSMA devices inside th CSMA containers
    // For CSMA, data rate is a channel attribute, not a device attribute!
    // CSMA doesn't allow one to mix devices on a channel 
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("100Mbps"));
    csma.SetChannelAttribute("Delay", TimeValue(NanoSeconds(6560)));

    NetDeviceContainer csmaDevices;
    csmaDevices = csma.Install(csmaNodes);

    // Install the protocol stack on the containers 
    InternetStackHelper stack;
    stack.Install(p2pNodes.Get(0));
    stack.Install(csmaNodes);

    // IP address for point to point nodes
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer p2pInterfaces;
    p2pInterfaces = address.Assign(p2pDevices);

    // IP address for CSMA devices (variable chain of devices)
    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer csmaInterfaces;
    csmaInterfaces = address.Assign(csmaDevices);

    // PORT number is 9 here
    UdpEchoServerHelper echoServer(9);

    ApplicationContainer serverApps = echoServer.Install(csmaNodes.Get(nCsma));
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(10.0));

    UdpEchoClientHelper echoClient(csmaInterfaces.GetAddress(nCsma), 9);
    echoClient.SetAttribute("MaxPackets", UintegerValue(nPackets));
    echoClient.SetAttribute("Interval", TimeValue(Seconds(1.0)));
    echoClient.SetAttribute("PacketSize", UintegerValue(1024));

    ApplicationContainer clientApps = echoClient.Install(p2pNodes.Get(0));
    clientApps.Start(Seconds(2.0));
    clientApps.Stop(Seconds(10.0));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    pointToPoint.EnablePcap("newnet", p2pNodes.Get(0)->GetId(), 0);
    csma.EnablePcap("newnet", csmaNodes.Get(nCsma)->GetId(), 0, false);
    csma.EnablePcap("newnet", csmaNodes.Get(nCsma-1)->GetId(), 0, false);

    Simulator::Run();
    Simulator::Destroy();
    return 0;
}



