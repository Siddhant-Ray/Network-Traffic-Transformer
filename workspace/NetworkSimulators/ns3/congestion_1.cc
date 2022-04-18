/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include <string>
#include <fstream>
#include <cstdlib>
#include <map>
#include <chrono>
#include <ctime>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/gnuplot.h"
#include "ns3/node.h"
#include "ns3/traffic-control-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/ipv4-header.h"
#include "ns3/random-variable-stream.h"
#include "ns3/rng-seed-manager.h"

// =================================================================
// Topology details :
//           _								 _
//		     |	H1------+		+------H4	 |
// 		     |			|		|			 |
//   Senders |	H2------R1------R2-----H5	 |	Receivers
// 		     |			|		|			 |
// 		     |	H3------+		+------H6	 |
//           _                               _
// =================================================================

/*   Referenced from https://github.com/pritam001/ns3-dumbell-topology-simulation

    H1(n0), H2(n1), H3(n2), H4(n3), H5(n4), H6(n5), R1(n6), R2(n7) :: n stands for node
	Dumbbell topology is used with 
	H1, H2, H3 on left side of dumbbell,
	H4, H5, H6 on right side of dumbbell,
	and routers R1 and R2 form the bridge of dumbbell.
	H1 is attached with TCP Reno agent.
	H2 is attached with TCP Westfood agent.
	H3 is attached with TCP Fack agent.
	Links:
	H1R1/H2R1/H3R1/H4R2/H5R2/H6R2: P2P with 100Mbps and 20ms.
	R1R2: (dumbbell bridge) P2P with 10Mbps and 50ms.
	packet size: 1.2KB.
	Number of packets decided by Bandwidth delay product:
	i.e. # packets = Bandwidth*Delay(in bits)
	Therefore, max # packets (HiRj) = 100Mbps*20ms = 2000000
	and max # packets (R1R2) = 10Mbps*50ms = 500000

*/

using namespace ns3;

typedef uint32_t uint;
#define ERROR 0.000001

NS_LOG_COMPONENT_DEFINE("congestion_1");

// Referenced from ns3 docs for attaching a callback trace at 
// runtime becasuse it doesn't exist at creation.

class APP: public Application {
	private:
		virtual void StartApplication(void);
		virtual void StopApplication(void);

		void ScheduleTx(void);
		void SendPacket(void);

		Ptr<Socket>     mSocket;
		Address         mPeer;
		uint32_t        mPacketSize;
		uint32_t        mNPackets;
		DataRate        mDataRate;
		EventId         mSendEvent;
		bool            mRunning;
		uint32_t        mPacketsSent;

	public:
		APP();
		virtual ~APP();

		void Setup(Ptr<Socket> socket, Address address, uint packetSize, uint nPackets, DataRate dataRate);
		void ChangeRate(DataRate newRate);
		void recv(int numBytesRcvd);

};

APP::APP(): mSocket(0),
		    mPeer(),
		    mPacketSize(0),
		    mNPackets(0),
		    mDataRate(0),
		    mSendEvent(),
		    mRunning(false),
		    mPacketsSent(0) {
}

APP::~APP() {
	mSocket = 0;
}

void APP::Setup(Ptr<Socket> socket, Address address, uint packetSize, uint nPackets, DataRate dataRate) {
	mSocket = socket;
	mPeer = address;
	mPacketSize = packetSize;
	mNPackets = nPackets;
	mDataRate = dataRate;
}

void APP::StartApplication() {
	mRunning = true;
	mPacketsSent = 0;
	mSocket->Bind();
	mSocket->Connect(mPeer);
	SendPacket();
}

void APP::StopApplication() {
	mRunning = false;
	if(mSendEvent.IsRunning()) {
		Simulator::Cancel(mSendEvent);
	}
	if(mSocket) {
		mSocket->Close();
	}
}

void APP::SendPacket() {
	Ptr<Packet> packet = Create<Packet>(mPacketSize);
    // FlowIdTag flowid;
	// flowid.SetFlowId(5);
    // packet->AddPacketTag(flowid);

	// !DEBUG
	/*double tVal = Simulator::Now().GetSeconds();
	AsciiTraceHelper asc;
	ofstream std::cout("outputs/congestion_1/size.txt");
	if(int(tVal)%20==0){
	std::cout << "packetsize" << Simulator::Now().GetSeconds() << "\t" << packet->GetSize() <<"\n";
	}*/
	
	mSocket->Send(packet);
	
	if(++mPacketsSent < mNPackets) {
		ScheduleTx();
	}
}

void APP::ScheduleTx() {
	if (mRunning) {
		Time tNext(Seconds(mPacketSize*8/static_cast<double>(mDataRate.GetBitRate())));
		mSendEvent = Simulator::Schedule(tNext, &APP::SendPacket, this);
        // DEBUG!
		/*double tVal = Simulator::Now().GetSeconds();
		if(int(tVal)%10==0)
		std::cout << "time:stamp" << Simulator::Now().GetSeconds() << "\t" << mPacketsSent << std::endl;
		std::cout << "time:stamp" << Simulator::Now().GetSeconds() << "\t" << mPacketSize << std::endl;*/
	}
}

void APP::ChangeRate(DataRate newrate) {
	mDataRate = newrate;
	return;
}

static void CwndChange(Ptr<OutputStreamWrapper> stream, double startTime, uint oldCwnd, uint newCwnd) {
	*stream->GetStream() << Simulator::Now().GetSeconds() - startTime << "\t" << newCwnd << std::endl;
}

// TraceSource defined in parent class ns3::QueueBase, bytes in current queue
void BytesInQueueTrace(Ptr<OutputStreamWrapper> stream, uint32_t oldVal, uint32_t newVal)
{
  *stream->GetStream() << Simulator::Now().GetSeconds()<< " " <<newVal<<std::endl;
}

// TraceSource defined in parent class ns3::QueueBase, packets in current queue
void PacketsInQueueTrace(Ptr<OutputStreamWrapper> stream, uint32_t oldVal, uint32_t newVal)
{
  *stream->GetStream() << Simulator::Now().GetSeconds()<< " " <<newVal<<std::endl;
}

// TraceSource for RxDrops 
static void PhyRxDrop(Ptr<OutputStreamWrapper> stream, Ptr<Queue<Packet>> queue, uint intfID, Ptr<const Packet>p)
{      
    std::size_t size = queue->GetNPackets();
    std::size_t num = queue->GetTotalDroppedPackets();
    *stream->GetStream() << "Rx drop at:, "<< Simulator::Now().GetSeconds()<< ", "
                         << "Queue size is currently, "<< size << ", "
                         << "Packets dropped till now, "<< num << ", ";

    FlowIdTag flowid;
    *stream->GetStream()<< "Flow id is, "<< p->PeekPacketTag(flowid) << ", ";
    *stream->GetStream()<< "Packet uid is, "<< p->GetUid() << ", ";
    *stream->GetStream()<< "Packet size is, "<< p->GetSize() << ", ";
	*stream->GetStream()<< "Interface id is, "<< intfID << "\n";
    

    /*p->Print(*stream->GetStream());
    *stream->GetStream() << "\n";
    p->PrintPacketTags(*stream->GetStream());
    *stream->GetStream() << "\n";*/

}

// TraceSource for TxDrops 
static void PhyTxDrop(Ptr<OutputStreamWrapper> stream, Ptr<const Packet>p)
{   
    // NS_LOG_INFO("TxDrop at "<<Simulator::Now().GetSeconds());
    *stream->GetStream() << "Tx drop at:, "<< Simulator::Now().GetSeconds()<< ", ";
    FlowIdTag flowid;
    *stream->GetStream()<< "Flow id is, "<< p->PeekPacketTag(flowid) << ", ";
    *stream->GetStream()<< "Packet uid is, "<< p->GetUid() << ", ";
    *stream->GetStream()<< "Packet size is, "<< p->GetSize() << "\n";
}

// TraceSource for Rx packets successfully
static void PhyRxEnd(Ptr<OutputStreamWrapper> stream, Ptr<Queue<Packet>> queue, uint intfID, Ptr<const Packet>p)
{   
    std::size_t size = queue->GetNPackets();
    *stream->GetStream() << "Rx received at:, "<< Simulator::Now().GetSeconds()<< ", "
                         << "Queue size is currently, "<< size << ", ";
    
    FlowIdTag flowid;
    *stream->GetStream()<< "Flow id is, "<< p->PeekPacketTag(flowid) << ", ";
    *stream->GetStream()<< "Packet uid is, "<< p->GetUid() << ", ";
    *stream->GetStream()<< "Packet size is, "<< p->GetSize() << ", ";
	*stream->GetStream()<< "Interface id is, "<< intfID << ", ";

	
	Ptr<Packet> copy = p->Copy();
	// Headers must be removed in the order they're present.
    PppHeader pppHeader;
	copy->RemoveHeader(pppHeader);
    Ipv4Header ipHeader;
    copy->RemoveHeader(ipHeader);
	*stream->GetStream() << "IP ID is, "<< ipHeader.GetIdentification() << ", ";
	*stream->GetStream() << "DSCP is, "<< ipHeader.GetDscp() << ", ";
	*stream->GetStream() << "ECN is, "<< ipHeader.GetEcn() << ", ";
	// *stream->GetStream() << "TTL is "<< ipHeader.GetTtl() << ", ";
	*stream->GetStream() << "Payload size is, "<< ipHeader.GetPayloadSize() << ", ";
	// *stream->GetStream() << "Protocol is "<< ipHeader.GetProtocol() << ", ";
	*stream->GetStream() << "Source IP is, "<< ipHeader.GetSource() << ", ";
	*stream->GetStream() << "Destination IP is, "<< ipHeader.GetDestination() << ", ";
	TcpHeader tcpHeader;
    copy->RemoveHeader(tcpHeader);
	*stream->GetStream() << "TCP source port is, "<< tcpHeader.GetSourcePort() << ", ";
	*stream->GetStream() << "TCP destination port is, "<< tcpHeader.GetDestinationPort() << ", ";
	*stream->GetStream() << "TCP sequence num is, "<< tcpHeader.GetSequenceNumber() << ", ";
	*stream->GetStream() << "TCP current window size is, "<< tcpHeader.GetWindowSize() << ", ";

   	p->Print(*stream->GetStream());
    *stream->GetStream() << "\n";
}

// TraceSource for Tx packets successfully
static void PhyTxEnd(Ptr<OutputStreamWrapper> stream, uint intfID, Ptr<const Packet>p)
{   
    // NS_LOG_INFO("TxDrop at "<<Simulator::Now().GetSeconds());
    *stream->GetStream() << "Tx sent at:, "<< Simulator::Now().GetSeconds()<< ", ";
    FlowIdTag flowid;
    *stream->GetStream()<< "Flow id is, "<< p->PeekPacketTag(flowid) << ", ";
    *stream->GetStream()<< "Packet uid is, "<< p->GetUid() << ", ";
    *stream->GetStream()<< "Packet size is, "<< p->GetSize() << ", ";
	*stream->GetStream()<< "Interface id is, "<< intfID << ",";

	Ptr<Packet> copy = p->Copy();
	// Headers must be removed in the order they're present.
    PppHeader pppHeader;
	copy->RemoveHeader(pppHeader);
    Ipv4Header ipHeader;
    copy->RemoveHeader(ipHeader);
	*stream->GetStream() << "IP ID is, "<< ipHeader.GetIdentification() << ", ";
	*stream->GetStream() << "DSCP is, "<< ipHeader.GetDscp() << ", ";
	*stream->GetStream() << "ECN is, "<< ipHeader.GetEcn() << ", ";
	// *stream->GetStream() << "TTL is "<< ipHeader.GetTtl() << ", ";
	*stream->GetStream() << "Payload size is, "<< ipHeader.GetPayloadSize() << ", ";
	// *stream->GetStream() << "Protocol is "<< ipHeader.GetProtocol() << ", ";
	*stream->GetStream() << "Source IP is, "<< ipHeader.GetSource() << ", ";
	*stream->GetStream() << "Destination IP is, "<< ipHeader.GetDestination() << ", ";
	TcpHeader tcpHeader;
    copy->RemoveHeader(tcpHeader);
	*stream->GetStream() << "TCP source port is, "<< tcpHeader.GetSourcePort() << ", ";
	*stream->GetStream() << "TCP destination port is, "<< tcpHeader.GetDestinationPort() << ", ";
	*stream->GetStream() << "TCP sequence num is, "<< tcpHeader.GetSequenceNumber() << ", ";
	*stream->GetStream() << "TCP current window size is, "<< tcpHeader.GetWindowSize() << ", ";

   	p->Print(*stream->GetStream());
    *stream->GetStream() << "\n";

}

std::map<uint, uint> mapDrop;
static void packetDrop(Ptr<OutputStreamWrapper> stream, double startTime, uint myId) {
	*stream->GetStream() << Simulator::Now().GetSeconds() - startTime << "\t" << std::endl;
	if(mapDrop.find(myId) == mapDrop.end()) {
		mapDrop[myId] = 0;
	}
	mapDrop[myId]++;
}

// Method to change rate during flow for later
void IncRate(Ptr<APP> app, DataRate rate) {
	app->ChangeRate(rate);
	return;
}

// Method to set the TCP CC method, add GetTypeID() for type of TCP protocol
Ptr<Socket> uniFlow(Address sinkAddress, 
					uint sinkPort, 
					std::string tcpVariant, 
					Ptr<Node> hostNode, 
					Ptr<Node> sinkNode, 
					double startTime, 
					double stopTime,
					uint packetSize,
					uint numPackets,
					std::string dataRate,
					double appStartTime,
					double appStopTime) {

	if(tcpVariant.compare("TcpNewReno") == 0) {
		Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpNewReno::GetTypeId()));
		Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(2500));
	} else if(tcpVariant.compare("TcpWestwood") == 0) {
		Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpWestwood::GetTypeId()));
		Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(2500));
	} else if(tcpVariant.compare("TcpVegas") == 0) {
		Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpVegas::GetTypeId()));
		Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1380));
	} else {
        fprintf(stdout,"Default CC protocol for TCP flows\n");
		// fprintf(stderr, "Invalid TCP version\n");
		// exit(EXIT_FAILURE);
	}

	// We introduce random behaviout at flow creation time, so that we don't have to worry when sending flows!

	// Check seed once!
	RngSeedManager::SetSeed(1);
	// std::cout<< RngSeedManager::GetSeed() << std::endl;

	// For packet size
	double minChangePack = 50.0;
	double maxChangePack = 2000.0;
	
	Ptr<UniformRandomVariable> changePack = CreateObject<UniformRandomVariable>();
	changePack->SetAttribute("Min", DoubleValue(minChangePack));
	changePack->SetAttribute("Max", DoubleValue(maxChangePack));
	uint incPackSize = changePack->GetInteger();
	std::cout<< "Old packet size "<< packetSize <<
	" New packet size " << incPackSize + packetSize << std::endl;

	// For num packets
	double minChangeNum = 50000000.0;
	double maxChangeNum = 75000000.0;
	
	Ptr<UniformRandomVariable> changeNum = CreateObject<UniformRandomVariable>();
	changeNum->SetAttribute("Min", DoubleValue(minChangeNum));
	changeNum->SetAttribute("Max", DoubleValue(maxChangeNum));
	uint incNum = changeNum->GetInteger();
	std::cout<< "Old num packs "<< numPackets <<
	" New num packs " << incNum + numPackets << std::endl;

	// For sending times
	double minTimeShift = 2.0;
	double maxTimeShift = 50.0;
	
	Ptr<UniformRandomVariable> timeShift = CreateObject<UniformRandomVariable>();
	timeShift->SetAttribute("Min", DoubleValue(minTimeShift));
	timeShift->SetAttribute("Max", DoubleValue(maxTimeShift));
	double shifted = timeShift->GetValue();
	std::cout<< "Flow is shifted by " << shifted <<std::endl;

	// TODO: Add randomness for datarate and queue size
	/*std::string rate = dataRate;

	if (!rate.empty()){
		rate.pop_back();
		rate.pop_back();
		rate.pop_back();
		rate.pop_back();
	}

	std::cout<< rate << std::endl;

	// For datarate
	double minRateChange = 20.0;
	double maxRateChange = 200.0;
	
	Ptr<UniformRandomVariable> cngrate = CreateObject<UniformRandomVariable>();
	cngrate->SetAttribute("Min", DoubleValue(minRateChange));
	cngrate->SetAttribute("Max", DoubleValue(maxRateChange));
	double changerateby = cngrate->GetValue();
	std::cout<< "Change rate by " << changerateby <<std::endl;*/
	
	
	PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), sinkPort));
	ApplicationContainer sinkApps = packetSinkHelper.Install(sinkNode);
	sinkApps.Start(Seconds(startTime+shifted));
	sinkApps.Stop(Seconds(stopTime+shifted));

	Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(hostNode, TcpSocketFactory::GetTypeId());


	Ptr<APP> app = CreateObject<APP>();
	app->Setup(ns3TcpSocket, sinkAddress, incPackSize + packetSize, incNum + numPackets, DataRate(dataRate));
	// std::cout<<packetSize<<std::endl;
	hostNode->AddApplication(app);
	app->SetStartTime(Seconds(appStartTime+shifted));
	app->SetStopTime(Seconds(appStopTime+shifted));

	return ns3TcpSocket;
}

void SingleFlow(bool pcap, std::string algo) {
	NS_LOG_INFO("Sending single flows from senders to receivers...");
	std::string rateHR = "100Mbps";
	std::string latencyHR = "20ms";
	std::string rateRR = "10Mbps";
	std::string latencyRR = "50ms";

	uint packetSize = 120;		// 1.2KB packet size
	uint queueSizeHR = (100000*20)/ packetSize;
	uint queueSizeRR = (10000*50)/ packetSize;
    
    std::string strqueueSizeHR = std::to_string(queueSizeHR);
    std::string strqueueSizeRR = std::to_string(queueSizeRR);
    std::string packetFlag = "p";

    if (packetFlag == "p"){
        strqueueSizeHR = strqueueSizeHR.append(packetFlag);
        strqueueSizeRR = strqueueSizeRR.append(packetFlag);
    }

    // Test 
    NS_LOG_INFO("Max Packets in the RR queue are ...");
    NS_LOG_INFO(strqueueSizeRR);
    NS_LOG_INFO("Max Packets in the HR queue are ...");
    NS_LOG_INFO(strqueueSizeHR);

	uint numSender = 6;
	double errorP = ERROR;

	// set droptail queue mode as packets i.e. to use maxpackets as queuesize metric not bytes
	// this can only be done with the StringValue in the new ns3, with adding "p" for packets
    Config::SetDefault("ns3::QueueBase::MaxSize", StringValue("1000p"));

    /* THIS DOES NOT WORK!!!!!!!
    Config::SetDefault("ns3::DropTailQueue::MaxPackets", UintegerValue(queuesize));
	*/

	//Creating channel without IP address
	PointToPointHelper p2pHR, p2pRR;
	/*
		SetDeviceAttribute: sets attributes of pointToPointNetDevice
		DataRate
		Address: MACAddress
		ReceiveErrorModel
		InterframeGap: The time to wait between packet (frame) transmissions
		TxQueue: A queue to use as the transmit queue in the device.
		SetChannelAttribute: sets attributes of pointToPointChannel
		Delay: Transmission delay through the channel
		SetQueue: sets attribute of a queue say droptailqueue
		Mode: Whether to use Bytes (see MaxBytes) or Packets (see MaxPackets) as the maximum queue size metric.
		MaxPackets: The maximum number of packets accepted by this DropTailQueue.
		MaxBytes: The maximum number of bytes accepted by this DropTailQueue.
	*/
	p2pHR.SetDeviceAttribute("DataRate", StringValue(rateHR));
	p2pHR.SetChannelAttribute("Delay", StringValue(latencyHR));
	// p2pHR.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue(strqueueSizeHR));
    p2pHR.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize(strqueueSizeHR)));
	p2pRR.SetDeviceAttribute("DataRate", StringValue(rateRR));
	p2pRR.SetChannelAttribute("Delay", StringValue(latencyRR));
	// p2pRR.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue(strqueueSizeRR));
    p2pRR.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize(strqueueSizeHR)));

    // Bottleneck link traffic control configuration
    uint32_t queueDiscSize = 1000;
    TrafficControlHelper tchRR;
    tchRR.SetRootQueueDisc("ns3::PfifoFastQueueDisc", "MaxSize",
                                    QueueSizeValue(QueueSize(QueueSizeUnit::PACKETS, queueDiscSize)));

    // Test queue size at the traffic level
    NS_LOG_INFO("Packets in the traffic level queue are....");
    NS_LOG_INFO(std::to_string(queueDiscSize));                                

	//Adding some errorrate
	/*
		Error rate model attributes
		ErrorUnit: The error unit
		ErrorRate: The error rate.
		RanVar: The decision variable attached to this error model.
	*/
	Ptr<RateErrorModel> em = CreateObjectWithAttributes<RateErrorModel> ("ErrorRate", DoubleValue(errorP));

	//Empty node containers
	NodeContainer routers, senders, receivers;
	//Create n nodes and append pointers to them to the end of this NodeContainer. 
	routers.Create(2);
	senders.Create(numSender);
	receivers.Create(numSender);

	/*
		p2pHelper.Install:
		This method creates a ns3::PointToPointChannel with the attributes configured 
		by PointToPointHelper::SetChannelAttribute, then, for each node in the input container,
		we create a ns3::PointToPointNetDevice with the requested attributes, 
		a queue for this ns3::NetDevice, and associate the resulting ns3::NetDevice 
		with the ns3::Node and ns3::PointToPointChannel.
	*/
	NetDeviceContainer routerDevices = p2pRR.Install(routers);
	//Empty netdevicecontatiners
	NetDeviceContainer leftRouterDevices, rightRouterDevices, senderDevices, receiverDevices;

    //Adding links
	NS_LOG_INFO("Adding links");
	for(uint i = 0; i < numSender; ++i) {
        // !DEBUG
        /* std::cout << "Sender node Id:" << senders.Get(i)->GetId()<<std::endl;
        std::cout << "Receiver node Id:" << receivers.Get(i)->GetId()<<std::endl;*/

		NetDeviceContainer cleft = p2pHR.Install(routers.Get(0), senders.Get(i));
		leftRouterDevices.Add(cleft.Get(0));
		senderDevices.Add(cleft.Get(1));
		cleft.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(em));

		NetDeviceContainer cright = p2pHR.Install(routers.Get(1), receivers.Get(i));
		rightRouterDevices.Add(cright.Get(0));
		receiverDevices.Add(cright.Get(1));
		cright.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(em));
	}

	//Install Internet Stack
	/*
		For each node in the input container, aggregate implementations of 
		the ns3::Ipv4, ns3::Ipv6, ns3::Udp, and, ns3::Tcp classes. 
	*/
	NS_LOG_INFO("Install internet stack");
	InternetStackHelper stack;
	stack.Install(routers);
	stack.Install(senders);
	stack.Install(receivers);

    QueueDiscContainer qdiscs;
    qdiscs = tchRR.Install(routerDevices);

	//Adding IP addresses
	NS_LOG_INFO("Adding IP addresses");
	Ipv4AddressHelper routerIP = Ipv4AddressHelper("10.3.0.0", "255.255.255.0");	//(network, mask)
	Ipv4AddressHelper senderIP = Ipv4AddressHelper("10.1.0.0", "255.255.255.0");
	Ipv4AddressHelper receiverIP = Ipv4AddressHelper("10.2.0.0", "255.255.255.0");
	

	Ipv4InterfaceContainer routerIFC, senderIFCs, receiverIFCs, leftRouterIFCs, rightRouterIFCs;

	//Assign IP addresses to the net devices specified in the container 
	//based on the current network prefix and address base
	routerIFC = routerIP.Assign(routerDevices);

	for(uint i = 0; i < numSender; ++i) {
		NetDeviceContainer senderDevice;
		senderDevice.Add(senderDevices.Get(i));
		senderDevice.Add(leftRouterDevices.Get(i));
		Ipv4InterfaceContainer senderIFC = senderIP.Assign(senderDevice);
		senderIFCs.Add(senderIFC.Get(0));
		leftRouterIFCs.Add(senderIFC.Get(1));
		//Increment the network number and reset the IP address counter 
		//to the base value provided in the SetBase method.
		senderIP.NewNetwork();

		NetDeviceContainer receiverDevice;
		receiverDevice.Add(receiverDevices.Get(i));
		receiverDevice.Add(rightRouterDevices.Get(i));
		Ipv4InterfaceContainer receiverIFC = receiverIP.Assign(receiverDevice);
		receiverIFCs.Add(receiverIFC.Get(0));
		rightRouterIFCs.Add(receiverIFC.Get(1));
		receiverIP.NewNetwork();
	}

    /* Add queue callback on RR queue 
    */
    AsciiTraceHelper ascii;
    uint i = 0;
    while(i < 2){
        
        Ptr<NetDeviceQueueInterface> interface = routerDevices.Get(i)->GetObject<NetDeviceQueueInterface>();
        Ptr<NetDeviceQueue> queueInterface = interface->GetTxQueue(0);
        Ptr<DynamicQueueLimits> queueLimits = StaticCast<DynamicQueueLimits>(queueInterface->GetQueueLimits());

        Ptr<Queue<Packet>> queue = StaticCast<PointToPointNetDevice>(routerDevices.Get(i))->GetQueue();
        // std::string byte_file_name = "outputs/congestion_2/bytesInQueue_router_" + std::to_string(0) + ".txt";
        Ptr<OutputStreamWrapper> streamBytesInQueue = ascii.CreateFileStream("outputs/congestion_1/bytesInQueue_router_"
                                                                             + std::to_string(i) + ".txt");
        queue->TraceConnectWithoutContext("BytesInQueue",MakeBoundCallback(&BytesInQueueTrace, streamBytesInQueue));

        // std::string packet_file_name = "outputs/congestion_2/packetsInQueue_router_" + std::to_string(0) + ".txt";
        Ptr<OutputStreamWrapper> streamPacketsInQueue = ascii.CreateFileStream("outputs/congestion_1/packetsInQueue_router_"
                                                                             + std::to_string(i) + ".txt");
        queue->TraceConnectWithoutContext("PacketsInQueue",MakeBoundCallback(&PacketsInQueueTrace, streamPacketsInQueue));
        i++;
    }

    uint j = 0;
    while (j < 2){
        Ptr<NetDevice> dev = routerDevices.Get(j);
        Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream("outputs/congestion_1/router" +
                                                                    std::to_string(j) + ".tr");
        p2pRR.EnableAscii(stream, dev);
        j++;
    }

    /*// !DEBUG
    std::cout<<"No of devices in router device container "<< routerDevices.GetN()<<std::endl;
    std::cout<< routerDevices.Get(0)<<std::endl;*/
    

    /*
		Measuring Performance of each TCP variant
	*/

	NS_LOG_INFO("Measuring Performance of each TCP variant...");
	/********************************************************************
		One flow for each host and measure
		1) Throughput for long durations
		2) Evolution of Congestion window
	********************************************************************/
	double durationGap = 1000;
	double oneFlowStart = 0;
	double otherFlowStart = 20;
    double netDuration = otherFlowStart + durationGap;
	uint port = 9000;
	uint numPackets = 100000000;
	std::string transferSpeed = "400Mbps";	
    std::string ccalgo = algo;

	//TCP NewReno from H1 to H4
	AsciiTraceHelper asciiTraceHelper;
	Ptr<OutputStreamWrapper> stream1CWND = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h1h7_singleflow.cwnd");
	Ptr<OutputStreamWrapper> stream1PD = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h1h7_singleflow.congestion_loss");
	
	Ptr<Socket> ns3TcpSocket1 = uniFlow(InetSocketAddress(receiverIFCs.GetAddress(0), port), port, ccalgo, senders.Get(0),
                                                        receivers.Get(0), oneFlowStart, oneFlowStart+durationGap, packetSize,
                                                         numPackets, transferSpeed, oneFlowStart, oneFlowStart+durationGap);
	ns3TcpSocket1->TraceConnectWithoutContext("CongestionWindow", MakeBoundCallback (&CwndChange, stream1CWND, 0));
	ns3TcpSocket1->TraceConnectWithoutContext("Drop", MakeBoundCallback (&packetDrop, stream1PD, 0, 1));

	// netDuration += durationGap;

	//TCP NewReno from H2 to H5
	Ptr<OutputStreamWrapper> stream2CWND = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h2h8_singleflow.cwnd");
	Ptr<OutputStreamWrapper> stream2PD = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h2h8_singleflow.congestion_loss");
	
	Ptr<Socket> ns3TcpSocket2 = uniFlow(InetSocketAddress(receiverIFCs.GetAddress(1), port), port, ccalgo, senders.Get(1),
                                                        receivers.Get(1), otherFlowStart, otherFlowStart+durationGap, packetSize,
                                                         numPackets, transferSpeed, otherFlowStart, otherFlowStart+durationGap);
	ns3TcpSocket2->TraceConnectWithoutContext("CongestionWindow", MakeBoundCallback(&CwndChange, stream2CWND, 0));
	ns3TcpSocket2->TraceConnectWithoutContext("Drop", MakeBoundCallback(&packetDrop, stream2PD, 0, 2));

	netDuration += durationGap;

	//TCP NewReno from H3 to H6
	Ptr<OutputStreamWrapper> stream3CWND = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h3h9_singleflow.cwnd");
	Ptr<OutputStreamWrapper> stream3PD = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h3h9_singleflow.congestion_loss");
	
	Ptr<Socket> ns3TcpSocket3 = uniFlow(InetSocketAddress(receiverIFCs.GetAddress(2), port), port, ccalgo, senders.Get(2),
                                                        receivers.Get(2), otherFlowStart, otherFlowStart+durationGap, packetSize,
                                                         numPackets, transferSpeed, otherFlowStart, otherFlowStart+durationGap);
	ns3TcpSocket3->TraceConnectWithoutContext("CongestionWindow", MakeBoundCallback(&CwndChange, stream3CWND, 0));
	ns3TcpSocket3->TraceConnectWithoutContext("Drop", MakeBoundCallback(&packetDrop, stream3PD, 0, 3));

	//TCP NewReno from H4 to H8
	Ptr<OutputStreamWrapper> stream4CWND = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h4h10_singleflow.cwnd");
	Ptr<OutputStreamWrapper> stream4PD = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h4h10_singleflow.congestion_loss");
	
	Ptr<Socket> ns3TcpSocket4 = uniFlow(InetSocketAddress(receiverIFCs.GetAddress(3), port), port, ccalgo, senders.Get(3),
                                                        receivers.Get(3), otherFlowStart, otherFlowStart+durationGap, packetSize,
                                                         numPackets, transferSpeed, otherFlowStart, otherFlowStart+durationGap);
	ns3TcpSocket4->TraceConnectWithoutContext("CongestionWindow", MakeBoundCallback(&CwndChange, stream4CWND, 0));
	ns3TcpSocket4->TraceConnectWithoutContext("Drop", MakeBoundCallback(&packetDrop, stream4PD, 0, 4));

	//TCP NewReno from H5 to H10
	Ptr<OutputStreamWrapper> stream5CWND = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h5h11_singleflow.cwnd");
	Ptr<OutputStreamWrapper> stream5PD = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h5h11_singleflow.congestion_loss");
	
	Ptr<Socket> ns3TcpSocket5 = uniFlow(InetSocketAddress(receiverIFCs.GetAddress(4), port), port, ccalgo, senders.Get(4),
                                                        receivers.Get(4), otherFlowStart, otherFlowStart+durationGap, packetSize,
                                                         numPackets, transferSpeed, otherFlowStart, otherFlowStart+durationGap);
	ns3TcpSocket5->TraceConnectWithoutContext("CongestionWindow", MakeBoundCallback(&CwndChange, stream5CWND, 0));
	ns3TcpSocket5->TraceConnectWithoutContext("Drop", MakeBoundCallback(&packetDrop, stream5PD, 0, 5));

	//TCP NewReno from H6 to H12
	Ptr<OutputStreamWrapper> stream6CWND = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h6h12_singleflow.cwnd");
	Ptr<OutputStreamWrapper> stream6PD = asciiTraceHelper.CreateFileStream("outputs/congestion_1/h6h12_singleflow.congestion_loss");
	
	Ptr<Socket> ns3TcpSocket6 = uniFlow(InetSocketAddress(receiverIFCs.GetAddress(5), port), port, ccalgo, senders.Get(5),
                                                        receivers.Get(5), otherFlowStart, otherFlowStart+durationGap, packetSize,
                                                         numPackets, transferSpeed, otherFlowStart, otherFlowStart+durationGap);
	ns3TcpSocket6->TraceConnectWithoutContext("CongestionWindow", MakeBoundCallback(&CwndChange, stream3CWND, 0));
	ns3TcpSocket6->TraceConnectWithoutContext("Drop", MakeBoundCallback(&packetDrop, stream3PD, 0, 6));

	

    uint routerNum = 0;
    while(routerNum <= 2){
    Ptr<Queue<Packet>> rqueue = StaticCast<PointToPointNetDevice>(routerDevices.Get(0))->GetQueue();

    // Log Rx drops on the router  
    Ptr<OutputStreamWrapper> streamRxDrops = ascii.CreateFileStream("outputs/congestion_1/RxDrops_lrouter_"
                                                                            + std::to_string(routerNum) + ".csv");
    leftRouterDevices.Get(routerNum)->TraceConnectWithoutContext("PhyRxDrop", MakeBoundCallback(&PhyRxDrop, streamRxDrops, rqueue,
                                                                routerNum));

    // Log Tx drops on the router 
    Ptr<OutputStreamWrapper> streamTxDrops = ascii.CreateFileStream("outputs/congestion_1/TxDrops_lrouter_"
                                                                            + std::to_string(routerNum) + ".csv");
    leftRouterDevices.Get(routerNum)->TraceConnectWithoutContext("PhyTxDrop", MakeBoundCallback(&PhyTxDrop, streamTxDrops));

    // Log Rx packets on router  
    Ptr<OutputStreamWrapper> streamRxEnds = ascii.CreateFileStream("outputs/congestion_1/RxRevd_lrouter_"
                                                                            + std::to_string(routerNum) + ".csv");
    leftRouterDevices.Get(routerNum)->TraceConnectWithoutContext("PhyRxEnd", MakeBoundCallback(&PhyRxEnd, streamRxEnds, rqueue,
                                                                routerNum));

    routerNum++;
    }

	// Log Tx packets sent from router , output interface is on different container so numSenders + 1, indexed from 0.
    Ptr<OutputStreamWrapper> rstreamTxEnds = ascii.CreateFileStream("outputs/congestion_1/TxSent_router_"
                                                                            + std::to_string(0) + ".csv");
    routerDevices.Get(0)->TraceConnectWithoutContext("PhyTxEnd", MakeBoundCallback(&PhyTxEnd, rstreamTxEnds,
                                                    1 + numSender -1));

    
    uint senderNum = 0;
	uint receiverNum = 0;
    while((senderNum < numSender) && (receiverNum < numSender)){

    // Log Tx packets sent from senders , this must be the same as packets received on router 0 (it is!!)
    Ptr<OutputStreamWrapper> streamSTxEnds = ascii.CreateFileStream("outputs/congestion_1/TxSent_sender_"
                                                                            + std::to_string(senderNum) + ".csv");
    senderDevices.Get(senderNum)->TraceConnectWithoutContext("PhyTxEnd", MakeBoundCallback(&PhyTxEnd, streamSTxEnds, 0));

    // Log Tx drops on the sender 0 
    Ptr<OutputStreamWrapper> streamSTxDrops = ascii.CreateFileStream("outputs/congestion_1/TxDrops_sender_"
                                                                            + std::to_string(senderNum) + ".csv");
    senderDevices.Get(senderNum)->TraceConnectWithoutContext("PhyTxDrop", MakeBoundCallback(&PhyTxDrop, streamSTxDrops));

    
	Ptr<Queue<Packet>> revqueue = StaticCast<PointToPointNetDevice>(receiverDevices.Get(receiverNum))->GetQueue();

	// Log Rx packets received on receivers , this must be the same as packets received on router 0 (it is!!)
    Ptr<OutputStreamWrapper> streamRRxEnds = ascii.CreateFileStream("outputs/congestion_1/RxRevd_receiver_"
                                                                            + std::to_string(receiverNum) + ".csv");
    receiverDevices.Get(receiverNum)->TraceConnectWithoutContext("PhyRxEnd", MakeBoundCallback(&PhyRxEnd, streamRRxEnds,
																	revqueue, 0));
	senderNum++;
	receiverNum++;
	}
	
    if (pcap)
    {
		
		p2pHR.EnablePcap("outputs/congestion_1/pcap/first_sender", senders.Get(0)->GetId(), 0);
		p2pHR.EnablePcap("outputs/congestion_1/pcap/second_sender", senders.Get(1)->GetId(), 0);
		p2pHR.EnablePcap("outputs/congestion_1/pcap/third_sender", senders.Get(2)->GetId(), 0);
		p2pHR.EnablePcap("outputs/congestion_1/pcap/fourth_sender", senders.Get(3)->GetId(), 0);
		p2pHR.EnablePcap("outputs/congestion_1/pcap/fifth_sender", senders.Get(4)->GetId(), 0);
		p2pHR.EnablePcap("outputs/congestion_1/pcap/sixth_sender", senders.Get(5)->GetId(), 0);
        
    }

	//Turning on Static Global Routing
	Ipv4GlobalRoutingHelper::PopulateRoutingTables();

	Ptr<FlowMonitor> flowmon;
	FlowMonitorHelper flowmonHelper;
	flowmon = flowmonHelper.InstallAll();
	// Run simulator for shifted flow durations (max possible shift)
	Simulator::Stop(Seconds(otherFlowStart+durationGap + 50));
	Simulator::Run();
	flowmon->CheckForLostPackets();

	//Ptr<OutputStreamWrapper> streamTP = asciiTraceHelper.CreateFileStream("application_1_a.tp");
	Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
	std::map<FlowId, FlowMonitor::FlowStats> stats = flowmon->GetFlowStats();
	for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i) {
		Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
		/* DEBUG!
		*streamTP->GetStream()  << "Flow " << i->first  << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n";
		*streamTP->GetStream()  << "  Tx Bytes:   " << i->second.txBytes << "\n";
		*streamTP->GetStream()  << "  Rx Bytes:   " << i->second.rxBytes << "\n";
		*streamTP->GetStream()  << "  Time        " << i->second.timeLastRxPacket.GetSeconds() - i->second.timeFirstTxPacket.GetSeconds() << "\n";
		*streamTP->GetStream()  << "  Throughput: " << i->second.rxBytes * 8.0 / (i->second.timeLastRxPacket.GetSeconds() - i->second.timeFirstTxPacket.GetSeconds())/1024/1024  << " Mbps\n";	
		*/
		if(t.sourceAddress == "10.1.0.1") {
			if(mapDrop.find(1)==mapDrop.end())
				mapDrop[1] = 0;
			*stream1PD->GetStream() << ccalgo << " Flow " << i->first  << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n";
			*stream1PD->GetStream()  << "Net Packet Lost: " << i->second.lostPackets << "\n";
			*stream1PD->GetStream()  << "Packet Lost due to buffer overflow: " << mapDrop[1] << "\n";
			*stream1PD->GetStream()  << "Packet Lost due to Congestion: " << i->second.lostPackets - mapDrop[1] << "\n";

		} else if(t.sourceAddress == "10.1.1.1") {
			if(mapDrop.find(2)==mapDrop.end())
				mapDrop[2] = 0;
			*stream2PD->GetStream() << ccalgo << " Flow " << i->first  << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n";
			*stream2PD->GetStream()  << "Net Packet Lost: " << i->second.lostPackets << "\n";
			*stream2PD->GetStream()  << "Packet Lost due to buffer overflow: " << mapDrop[2] << "\n";
			*stream2PD->GetStream()  << "Packet Lost due to Congestion: " << i->second.lostPackets - mapDrop[2] << "\n";

		} else if(t.sourceAddress == "10.1.2.1") {
			if(mapDrop.find(3)==mapDrop.end())
				mapDrop[3] = 0;
			*stream3PD->GetStream() << ccalgo << " Flow " << i->first  << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n";
			*stream3PD->GetStream()  << "Net Packet Lost: " << i->second.lostPackets << "\n";
			*stream3PD->GetStream()  << "Packet Lost due to buffer overflow: " << mapDrop[3] << "\n";
			*stream3PD->GetStream()  << "Packet Lost due to Congestion: " << i->second.lostPackets - mapDrop[3] << "\n";

		}
	}

	//flowmon->SerializeToXmlFile("application_1_a.flowmon", true, true);
	NS_LOG_INFO("Simulation finished");
	Simulator::Destroy();

}

int main(int argc, char **argv) {

	bool pcap = true;

    CommandLine cmd;
    cmd.Parse (argc, argv);
    std::string ccalgo = "TcpVegas";

    auto start = std::chrono::system_clock::now();
	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	
	std::cout << "Started computation at " << std::ctime(&start_time);

	SingleFlow(pcap, ccalgo);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "Finished computation at " << std::ctime(&end_time)
              << "Elapsed time: " << elapsed_seconds.count() << "s\n";
}





